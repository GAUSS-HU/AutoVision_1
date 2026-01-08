import argparse
import yaml
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        in_features = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_image(img_path: str, image_size: int):
    tf = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0)  # (1, C, H, W)


@torch.no_grad()
def predict(image_path, model, classes, device, image_size, topk=5):
    x = load_image(image_path, image_size).to(device)
    model.eval()

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    topk_probs, topk_idx = torch.topk(probs, k=topk)
    results = [
        (classes[i], float(p))
        for i, p in zip(topk_idx.cpu().numpy(), topk_probs.cpu().numpy())
    ]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg["device"])

    # Food-101 类名文件（torchvision 自带）
    classes_path = os.path.join(cfg["data_dir"], "food-101", "meta", "classes.txt")
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    model = build_model(cfg["model"], int(cfg["num_classes"]))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    results = predict(
        image_path=args.image,
        model=model,
        classes=classes,
        device=device,
        image_size=int(cfg["image_size"]),
        topk=args.topk
    )

    print(f"\nImage: {args.image}")
    print("Prediction (Top-{}):".format(args.topk))
    for i, (cls, p) in enumerate(results, 1):
        print(f"{i}. {cls:20s}  prob={p:.4f}")


if __name__ == "__main__":
    main()
