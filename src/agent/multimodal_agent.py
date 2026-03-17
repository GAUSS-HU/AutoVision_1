import torch
import clip
from PIL import Image

# -------------------------
# 全局加载 CLIP（只加载一次）
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


@torch.no_grad()
def clip_verify(image_path, candidate_classes):
    """
    image_path: str
    candidate_classes: List[str]
    return: similarity scores (numpy array)
    """
    image = clip_preprocess(
        Image.open(image_path).convert("RGB")
    ).unsqueeze(0).to(device)

    texts = [f"a photo of {c}" for c in candidate_classes]
    text_tokens = clip.tokenize(texts).to(device)

    image_features = clip_model.encode_image(image)
    text_features = clip_model.encode_text(text_tokens)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).squeeze(0)
    return similarity.cpu().numpy()


def multimodal_decision_agent(
    image_path,
    cls_results,
    prob_gap=0.25,          # 差值门槛：差值足够大就不触发 CLIP
    min_conf=0.55,          # 低置信门槛：很不确定才考虑 CLIP
    topk_clip=3,
    clip_override_margin=0.10  # CLIP 要“明显更强”才允许推翻 Top-1
):
    """
    cls_results: [(class_name, prob), ...]
    return: final_class, reason
    """
    p1 = float(cls_results[0][1])
    p2 = float(cls_results[1][1])
    top1 = cls_results[0][0]

    gap = p1 - p2

    # 1) 差值足够大：直接信主模型（不管 p1 多高）
    if gap >= prob_gap:
        return top1, f"single-model confident (gap={gap:.3f})"

    # 2) 置信度还行：也先信主模型
    if p1 >= min_conf:
        return top1, f"single-model confident (p1={p1:.3f})"

    # 3) 只有在“又近又低”的情况下才调用 CLIP
    candidates = [c for c, _ in cls_results[:topk_clip]]
    clip_scores = clip_verify(image_path, candidates)

    # 4) 除非它对某个候选明显更偏好，否则不推翻 top1
    best_idx = int(clip_scores.argmax())
    best_cls = candidates[best_idx]

    # 将 clip_scores 做成相对差（只在候选内部比较）
    # 取 best 和 top1 在候选中的差距
    top1_idx = candidates.index(top1)
    margin = float(clip_scores[best_idx] - clip_scores[top1_idx])

    if best_cls != top1 and margin < clip_override_margin:
        return top1, f"clip checked, kept top1 (clip_margin={margin:.3f})"

    return best_cls, f"multi-model verified (clip_margin={margin:.3f})"

