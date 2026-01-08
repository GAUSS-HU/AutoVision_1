import os
import json
import uuid
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# 你的 Agent
from agent.multimodal_agent import multimodal_decision_agent

# ---- 重要：OpenMP 冲突处理（你之前遇到的那个）----
# 放在 import clip / torch 之后也行，但最好在最早处确保有值
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        in_features = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_features, num_classes)
        return m
    raise ValueError(f"Unknown model: {model_name}")


def load_classes(classes_path: str) -> List[str]:
    with open(classes_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def get_preprocess(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])


class InferenceService:
    """
    负责：
    - 模型加载（只加载一次）
    - 单图预测 Top-K
    - Agent 决策（必要时调用 CLIP）
    - 营养库查询
    """
    def __init__(
        self,
        ckpt_path: str,
        data_dir: str,
        nutrition_json_path: str,
        model_name: str = "convnext_tiny",
        num_classes: int = 101,
        image_size: int = 224,
        device: str = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.image_size = int(image_size)
        self.preprocess = get_preprocess(self.image_size)

        # Food-101 类名
        classes_path = os.path.join(data_dir, "food-101", "meta", "classes.txt")
        self.classes = load_classes(classes_path)

        # 模型
        self.model = build_model(model_name, int(num_classes))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # 兼容你保存的格式：ckpt["model_state"]
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # 营养库
        with open(nutrition_json_path, "r", encoding="utf-8") as f:
            self.nutrition_db = json.load(f)

    @torch.no_grad()
    def predict_topk(self, image_path: str, topk: int = 5) -> List[Tuple[str, float]]:
        img = Image.open(image_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topk_probs, topk_idx = torch.topk(probs, k=topk)
        results = [(self.classes[int(i)], float(p)) for i, p in zip(topk_idx.cpu(), topk_probs.cpu())]
        return results

    def run(self, image_path: str, topk: int = 5) -> Dict[str, Any]:
        # 1) 主模型 Top-K
        results = self.predict_topk(image_path, topk=topk)

        # 2) Agent 决策（可能调用 CLIP）
        final_class, agent_reason = multimodal_decision_agent(
            image_path=image_path,
            cls_results=results
        )

        # 3) 营养查询
        nutrition = self.nutrition_db.get(final_class, None)

        return {
            "topk": results,
            "final_class": final_class,
            "agent_reason": agent_reason,
            "nutrition": nutrition
        }


def save_upload(file_bytes: bytes, original_filename: str, out_dir: str = "outputs_web") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(original_filename)[1].lower() or ".jpg"
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path
