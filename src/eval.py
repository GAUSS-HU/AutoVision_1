import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import yaml
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    # eval时 pretrained=False，因为我们会load checkpoint
    if model_name == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        in_features = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_features, num_classes)
        return m
    elif model_name == "resnet50":
        m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_y = []
    all_pred = []
    all_conf = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        conf, pred = probs.max(dim=1)
        all_y.append(y.numpy())
        all_pred.append(pred.cpu().numpy())
        all_conf.append(conf.cpu().numpy())

    return np.concatenate(all_y), np.concatenate(all_pred), np.concatenate(all_conf)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str, max_classes: int = 30):
    """
    Food101 有101类，直接画会很密。
    这里默认只画前 max_classes 个类别的子矩阵（可改）。
    """
    k = min(max_classes, cm.shape[0])
    cm_small = cm[:k, :k]

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm_small, interpolation="nearest")
    plt.title(f"Confusion Matrix (top {k} classes)")
    plt.colorbar()
    tick_marks = np.arange(k)
    plt.xticks(tick_marks, class_names[:k], rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names[:k], fontsize=6)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_top_errors(dataset: datasets.VisionDataset, y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray,
                    class_names: List[str], out_dir: str, topk: int = 24):
    """
    保存 topk 个“高置信错误”样本（最危险的错）
    """
    os.makedirs(out_dir, exist_ok=True)
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print("No errors found.")
        return

    # 按错误置信度排序，挑最自信但错的
    wrong_sorted = wrong_idx[np.argsort(-conf[wrong_idx])]
    pick = wrong_sorted[:topk]

    # 画网格图
    cols = 6
    rows = int(np.ceil(len(pick) / cols))
    fig = plt.figure(figsize=(18, 3 * rows))

    for i, idx in enumerate(pick, start=1):
        img, _ = dataset[idx]  # dataset transform 后是 tensor
        # 反归一化用于可视化
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img_np = np.clip(img_np, 0, 1)

        t = class_names[y_true[idx]]
        p = class_names[y_pred[idx]]
        c = conf[idx]

        ax = plt.subplot(rows, cols, i)
        ax.imshow(img_np)
        ax.axis("off")
        ax.set_title(f"T:{t}\nP:{p}\nconf={c:.2f}", fontsize=9)

    out_path = os.path.join(out_dir, "top_confident_errors.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top errors grid to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)  # outputs/.../best.pt
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg["device"])
    image_size = int(cfg["image_size"])

    val_tf = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    data_dir = cfg["data_dir"]
    val_ds = datasets.Food101(root=data_dir, split="test", download=False, transform=val_tf)

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg["model"], int(cfg["num_classes"]), pretrained=False)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    y_true, y_pred, conf = predict_all(model, val_loader, device)

    acc = float((y_true == y_pred).mean())
    print(f"Accuracy: {acc:.4f}")

    # 文本报告（会很长，保存到文件）
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    report = classification_report(y_true, y_pred, target_names=val_ds.classes, digits=4)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved classification report to {os.path.join(out_dir, 'classification_report.txt')}")

    cm = confusion_matrix(y_true, y_pred)
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)

    plot_confusion_matrix(cm, val_ds.classes, os.path.join(out_dir, "confusion_matrix_top30.png"), max_classes=30)
    print(f"Saved confusion matrix plot to {os.path.join(out_dir, 'confusion_matrix_top30.png')}")

    save_top_errors(val_ds, y_true, y_pred, conf, val_ds.classes, out_dir, topk=24)


if __name__ == "__main__":
    main()
