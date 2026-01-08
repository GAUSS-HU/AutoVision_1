import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import yaml
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json




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

def draw_predictions_with_nutrition(
    image_path,
    results,
    nutrition,
    save_path,
    font_size=60,
    margin=10,
    nutrition_msg=None
):
    img = Image.open(image_path).convert("RGB")
    img_rgba = img.convert("RGBA")
    draw = ImageDraw.Draw(img_rgba)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # ---------- 组织文本 ----------
    lines = ["Prediction:"]
    for i, (cls, p) in enumerate(results, 1):
        lines.append(f"{i}. {cls} ({p:.2f})")

    lines.append("")
    lines.append("Nutrition:")

    if nutrition is not None:
        # nutrition 是 dict
        lines.append(f"Calories: {nutrition.get('calories_kcal', 'N/A')} kcal")
        lines.append(f"Protein: {nutrition.get('protein_g', 'N/A')} g")
        lines.append(f"Fat: {nutrition.get('fat_g', 'N/A')} g")
        lines.append(f"Carbs: {nutrition.get('carbs_g', 'N/A')} g")
    else:
        # 无营养数据
        lines.append(nutrition_msg or "No nutrition data available.")

    # ---------- 计算背景框大小 ----------
    text_w = max(draw.textlength(line, font=font) for line in lines if line)
    text_h = font_size * len(lines) + margin * 2

    bg = Image.new("RGBA", (int(text_w + margin * 2), int(text_h)), (0, 0, 0, 160))
    img_rgba.paste(bg, (0, 0), bg)

    # ---------- 写字 ----------
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill=(255, 255, 255, 255), font=font)
        y += font_size

    img_rgba.convert("RGB").save(save_path)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open("data/nutrition/food101_nutrition.json", "r", encoding="utf-8") as f:
        nutrition_db = json.load(f)



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

    food = results[0][0]  # Top-1 类别
    nutrition = nutrition_db.get(food, None)
    if nutrition:
        print("\nNutrition (approx.):")
        for k, v in nutrition.items():
            print(f"{k}: {v}")
    else:
        print("\nNo nutrition data available.")

    print(f"\nImage: {args.image}")
    print("Prediction (Top-{}):".format(args.topk))
    for i, (cls, p) in enumerate(results, 1):
        print(f"{i}. {cls:20s}  prob={p:.4f}")

        # 保存带预测结果的图片
        out_img = os.path.splitext(args.image)[0] + "_pred.png"

        draw_predictions_with_nutrition(
            image_path=args.image,
            results=results,
            nutrition=nutrition,
            save_path=out_img,
            font_size=60
        )

        print(f"\nSaved visualization to: {out_img}")


if __name__ == "__main__":
    main()
