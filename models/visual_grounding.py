"""
Vision Grounding Module for Semantic VLA.

Pipeline:
  RGB Image(s)
      ↓
  OWL-ViT zero-shot object detection (text prompts from task instruction)
      ↓
  Detected boxes + labels + scores
      ↓
  Map labels → ObjectCat / TargetCat
      ↓
  Build VisualSemanticRepresentation
      ↓
  Encode → Visual Semantic Embedding → fuse with AbstractTask embedding
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBJECT_PROMPTS = {
    "graspable_container": [
        "a mug", "a cup", "a bottle", "a wine bottle", "a moka pot",
        "a casserole dish", "a water bottle",
    ],
    "bowl_like": [
        "a bowl", "a ramekin", "a small bowl",
    ],
    "box_container": [
        "a basket", "a box", "a cookie box", "a storage box",
        "a caddy", "a tray",
    ],
    "food_item": [
        "a can of soup", "a soup can", "cream cheese", "a butter block",
        "a bottle of ketchup", "a bottle of BBQ sauce", "a bottle of salad dressing",
        "a bottle of tomato sauce", "a milk carton", "a juice carton",
        "a chocolate pudding cup", "a pudding cup",
    ],
    "flat_surface": [
        "a plate", "a tray", "a cutting board", "a stove", "a table",
        "a counter top", "a rack", "a cabinet top",
    ],
    "articulated": [
        "a drawer", "a cabinet", "a microwave", "an oven", "a door",
    ],
}

ALL_PROMPTS = []
for prompts in OBJECT_PROMPTS.values():
    ALL_PROMPTS.extend(prompts)

PROMPT_TO_CATEGORY = {}
for cat, prompts in OBJECT_PROMPTS.items():
    for p in prompts:
        PROMPT_TO_CATEGORY[p] = cat

@dataclass
class DetectedObject:
    category: str
    bbox: List[float]
    confidence: float
    label_text: str

@dataclass
class VisualSemantic:
    objects: List[DetectedObject] = field(default_factory=list)
    num_graspable_containers: int = 0
    num_bowl_like: int = 0
    num_food_items: int = 0
    num_box_containers: int = 0
    num_flat_surfaces: int = 0
    num_articulated: int = 0
    image_width: int = 0
    image_height: int = 0
    step_idx: int = 0
    total_steps: int = 800
    prev_objects: Optional[List[DetectedObject]] = None

    def get_count(self, category: str) -> int:
        return sum(1 for o in self.objects if o.category == category)

    def has_category(self, category: str) -> bool:
        return self.get_count(category) > 0

    def get_step_progress(self) -> float:
        return min(self.step_idx / max(self.total_steps, 1), 1.0)

    def get_change_stats(self) -> Tuple[int, int, int]:
        if self.prev_objects is None or len(self.prev_objects) == 0:
            return (0, 0, 0)
        curr_counts = {}
        for o in self.objects:
            curr_counts[o.category] = curr_counts.get(o.category, 0) + 1
        prev_counts = {}
        for o in self.prev_objects:
            prev_counts[o.category] = prev_counts.get(o.category, 0) + 1
        removed = 0
        added = 0
        for cat in set(list(curr_counts.keys()) + list(prev_counts.keys())):
            diff = curr_counts.get(cat, 0) - prev_counts.get(cat, 0)
            if diff < 0:
                removed += abs(diff)
            elif diff > 0:
                added += diff
        moved = 0
        for co in self.objects:
            for po in self.prev_objects:
                if co.category == po.category:
                    cx = (co.bbox[0] + co.bbox[2]) / 2
                    cy = (co.bbox[1] + co.bbox[3]) / 2
                    px = (po.bbox[0] + po.bbox[2]) / 2
                    py = (po.bbox[1] + po.bbox[3]) / 2
                    dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if dist > 0.15:
                        moved += 1
        return (removed, added, moved)

def load_owl_vit_pipeline(model_name: str = "google/owlvit-base-patch32", device: str = "cuda"):
    from transformers import pipeline
    print(f"Loading OWL-ViT pipeline: {model_name} ...")
    pipe = pipeline(
        "zero-shot-object-detection",
        model=model_name,
        device=device if torch.cuda.is_available() else -1,
    )
    print(f"  OWL-ViT ready on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    return pipe

def detect_objects(
    pipe,
    image: np.ndarray,
    text_prompts: Optional[List[str]] = None,
    confidence_threshold: float = 0.05,
    nms_threshold: float = 0.5,
) -> List[DetectedObject]:
    if text_prompts is None:
        text_prompts = ALL_PROMPTS

    h, w = image.shape[:2]

    from PIL import Image as _PILImage
    pil_img = _PILImage.fromarray(image)

    results = pipe(
        image=pil_img,
        candidate_labels=text_prompts,
        threshold=confidence_threshold,
    )

    detections = []
    for r in results:
        label = r["label"]
        score = r["score"]
        bbox = [r["box"]["xmin"], r["box"]["ymin"], r["box"]["xmax"], r["box"]["ymax"]]

        cat = PROMPT_TO_CATEGORY.get(label, "unknown")
        det = DetectedObject(
            category=cat,
            bbox=[b / w if i % 2 == 0 else b / h for i, b in enumerate(bbox)],
            confidence=score,
            label_text=label,
        )
        detections.append(det)

    detections = _nms_by_category(detections, nms_threshold)
    return detections

def _nms_by_category(detections: List[DetectedObject], threshold: float) -> List[DetectedObject]:
    by_cat: Dict[str, List[DetectedObject]] = {}
    for d in detections:
        by_cat.setdefault(d.category, []).append(d)

    kept: List[DetectedObject] = []
    for cat, dets in by_cat.items():
        dets.sort(key=lambda d: -d.confidence)
        ious = _compute_ious([d.bbox for d in dets])
        keep = []
        for i, d in enumerate(dets):
            if all(ious[i][j] < threshold for j in keep):
                keep.append(i)
        kept.extend(dets[i] for i in keep)
    return kept

def _compute_ious(boxes: List[List[float]]) -> np.ndarray:
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 0))
    arr = np.array(boxes)
    x1 = arr[:, 0][:, None]
    y1 = arr[:, 1][:, None]
    x2 = arr[:, 2][:, None]
    y2 = arr[:, 3][:, None]
    inter_x1 = np.maximum(x1, x1.T)
    inter_y1 = np.maximum(y1, y1.T)
    inter_x2 = np.minimum(x2, x2.T)
    inter_y2 = np.minimum(y2, y2.T)
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area = (x2 - x1) * (y2 - y1)
    union = area + area.T - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou

def build_visual_semantic(
    detections: List[DetectedObject],
    img_h: int,
    img_w: int,
    step_idx: int = 0,
    total_steps: int = 800,
    prev_vs: Optional["VisualSemantic"] = None,
) -> VisualSemantic:
    vs = VisualSemantic(objects=detections, image_width=img_w, image_height=img_h)
    vs.num_graspable_containers = vs.get_count("graspable_container")
    vs.num_bowl_like = vs.get_count("bowl_like")
    vs.num_food_items = vs.get_count("food_item")
    vs.num_box_containers = vs.get_count("box_container")
    vs.num_flat_surfaces = vs.get_count("flat_surface")
    vs.num_articulated = vs.get_count("articulated")
    vs.step_idx = step_idx
    vs.total_steps = total_steps
    vs.prev_objects = prev_vs.objects if prev_vs is not None else None
    return vs

class VisualSemanticEncoder(nn.Module):
    """
    Encode VisualSemantic → multi-token sequence [B, max_objects+1, hidden].

    Token 0 = global token: count_emb + α_step(0-init) * step_emb + α_change(0-init) * change_emb
    Tokens 1..N = per-object: cat_emb + bbox_emb (original V3, NOT modified!)

    零初始化门控: 新约束的门控参数初始化为 0 → 训练初期完全等价于 V3！
    """

    NUM_CATEGORIES = 7
    MAX_OBJECTS = 8

    def __init__(self, hidden_size: int = 256, max_objects: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_objects = max_objects

        self.category_embed = nn.Embedding(self.NUM_CATEGORIES + 1, hidden_size)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.count_mlp = nn.Sequential(
            nn.Linear(self.NUM_CATEGORIES, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.object_fuse = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

        self.step_mlp = nn.Sequential(
            nn.Linear(1, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )
        self.change_mlp = nn.Sequential(
            nn.Linear(3, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )
        self.alpha_step = nn.Parameter(torch.zeros(1))
        self.alpha_change = nn.Parameter(torch.zeros(1))

    def forward(self, visual_semantics: List[VisualSemantic]) -> torch.Tensor:
        B = len(visual_semantics)
        device = self.category_embed.weight.device
        dtype = self.category_embed.weight.dtype

        cat_id_map = {
            "graspable_container": 0,
            "bowl_like": 1,
            "food_item": 2,
            "box_container": 3,
            "flat_surface": 4,
            "articulated": 5,
            "unknown": 6,
        }

        max_tokens = self.max_objects + 1
        padded = torch.zeros(B, max_tokens, self.hidden_size, device=device, dtype=dtype)
        mask = torch.zeros(B, max_tokens, device=device, dtype=torch.long)

        for b, vs in enumerate(visual_semantics):
            counts = torch.zeros(self.NUM_CATEGORIES, device=device, dtype=dtype)
            for obj in vs.objects[: self.max_objects]:
                cid = cat_id_map.get(obj.category, 6)
                if cid < self.NUM_CATEGORIES:
                    counts[cid] += 1.0
            count_emb = self.count_mlp(counts)

            progress = vs.get_step_progress()
            step_val = torch.tensor([progress], device=device, dtype=dtype)
            step_emb = self.step_mlp(step_val)

            rem, add, mov = vs.get_change_stats()
            change_val = torch.tensor([rem, add, mov], device=device, dtype=dtype).float()
            change_emb = self.change_mlp(change_val)

            padded[b, 0] = (
                count_emb
                + self.alpha_step * step_emb
                + self.alpha_change * change_emb
            )
            mask[b, 0] = 1

            sorted_objs = sorted(
                vs.objects[: self.max_objects],
                key=lambda o: (o.bbox[0] + o.bbox[2]) / 2,
            )

            for i, obj in enumerate(sorted_objs):
                cid = cat_id_map.get(obj.category, 6)
                cat_emb = self.category_embed(torch.tensor(cid, device=device))
                bbox = torch.tensor(obj.bbox, device=device, dtype=dtype)
                bbox_emb = self.bbox_mlp(bbox)
                padded[b, i + 1] = self.object_fuse(torch.cat([cat_emb, bbox_emb]))
                mask[b, i + 1] = 1

        self._last_mask = mask
        return padded

def visualize_detections(image: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
    import cv2
    vis = image.copy()
    h, w = vis.shape[:2]
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255),
    ]
    cat_color = {}
    for d in detections:
        if d.category not in cat_color:
            cat_color[d.category] = colors[len(cat_color) % len(colors)]
        c = cat_color[d.category]
        x1, y1, x2, y2 = [
            int(d.bbox[0] * w), int(d.bbox[1] * h),
            int(d.bbox[2] * w), int(d.bbox[3] * h),
        ]
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        cv2.putText(
            vis, f"{d.category}: {d.confidence:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1,
        )
    return vis

if __name__ == "__main__":
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dets = [
        DetectedObject(category="graspable_container", bbox=[0.1, 0.1, 0.3, 0.4], confidence=0.9, label_text="a mug"),
        DetectedObject(category="graspable_container", bbox=[0.5, 0.2, 0.7, 0.5], confidence=0.8, label_text="a moka pot"),
        DetectedObject(category="food_item", bbox=[0.2, 0.6, 0.4, 0.8], confidence=0.7, label_text="cream cheese"),
    ]
    vs = build_visual_semantic(dets, dummy_img.shape[0], dummy_img.shape[1], step_idx=15, total_steps=40)
    print(f"\nImage: {dummy_img.shape}, {len(dets)} detections")
    print(f"  step_progress={vs.get_step_progress():.2f}")
    print(f"  change_stats={vs.get_change_stats()}")

    enc = VisualSemanticEncoder(hidden_size=256)
    out = enc([vs])
    print(f"\nVisualSemanticEncoder output: {out.shape}")
    print(f"  alpha_step={torch.sigmoid(enc.alpha_step).item():.4f} (init=0 → 0.5)")
    print(f"  alpha_change={torch.sigmoid(enc.alpha_change).item():.4f} (init=0 → 0.5)")
