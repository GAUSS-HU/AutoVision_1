import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import os, yaml, time, random
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


def rand_bbox(W, H, lam):
    # 根据 lam 计算裁剪框大小（面积比例 ~ 1-lam）
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 随机中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def apply_cutmix(x, y, alpha=1.0):
    """
    x: (B, C, H, W), y: (B,)
    return: mixed_x, y_a, y_b, lam
    """
    B, C, H, W = x.size()
    # 随机打乱 batch
    index = torch.randperm(B, device=x.device)

    y_a = y
    y_b = y[index]

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    x1, y1, x2, y2 = rand_bbox(W, H, lam)

    # 把别的图的块贴过来
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # 用真实贴图面积更新 lam（更精确）
    area = (x2 - x1) * (y2 - y1)
    lam = 1.0 - area / float(W * H)

    return x, y_a, y_b, lam


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_features, num_classes)
        return m
    elif model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item()) * x.size(0)

        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    return {
        "val_loss": loss_sum / max(total, 1),
        "val_acc": correct / max(total, 1),
    }


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Transforms
    image_size = int(cfg["image_size"])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Dataset
    data_dir = cfg["data_dir"]
    train_ds = datasets.Food101(root=data_dir, split="train", download=True, transform=train_tf)
    val_ds = datasets.Food101(root=data_dir, split="test", download=True, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = build_model(cfg["model"], int(cfg["num_classes"]), bool(cfg["pretrained"]))
    model.to(device)

    # Loss / Optim
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    if cfg["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    else:
        raise ValueError("Only adamw implemented for now")

    # ===== AMP mixed precision =====
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = -1.0
    history = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                cutmix_prob = float(cfg.get("cutmix_prob", 0.0))
                cutmix_alpha = float(cfg.get("cutmix_alpha", 1.0))

                use_cutmix = (np.random.rand() < cutmix_prob)

                if use_cutmix:
                    x, y_a, y_b, lam = apply_cutmix(x, y, alpha=cutmix_alpha)

                logits = model(x)

                if use_cutmix:
                    loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
                else:
                    loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))

            pbar.set_postfix({
                "loss": running_loss / max(total, 1),
                "acc": correct / max(total, 1),
                "device": str(device),
            })

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **val_metrics
        }
        history.append(row)
        print("Metrics:", row)

        # Save best
        if val_metrics["val_acc"] > best_acc:
            best_acc = val_metrics["val_acc"]
            ckpt_path = os.path.join(cfg["output_dir"], "best.pt")
            torch.save({
                "cfg": cfg,
                "model_state": model.state_dict(),
                "best_acc": best_acc,
                "epoch": epoch,
            }, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path} (acc={best_acc:.4f})")

    # Save last + history
    torch.save({"cfg": cfg, "model_state": model.state_dict()}, os.path.join(cfg["output_dir"], "last.pt"))
    with open(os.path.join(cfg["output_dir"], "history.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(history, f, allow_unicode=True)

    print(f"Done. Best val_acc={best_acc:.4f}. Outputs in {cfg['output_dir']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
