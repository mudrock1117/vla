"""
MVP Semantic Abstraction Schema (v1 - 精简版).

5 object categories + 3 target categories + 2 actions + 3 relations.
覆盖 LIBERO 90 任务中 > 90% 的操作模式。

Pipeline:
  "Pick up the red mug and place it on the tray"
    → AbstractTask(
        operation="pick_and_place",
        object_category="graspable_container",
        target_category="flat_surface_target",
        relations=["on"]
      )

设计原则:
  1. Category 之间**没有重叠** (每个物体只能映射到一个)
  2. 每个 category 有明确的 affordance 语义
  3. 从 specific → abstract 是 many-to-one 映射
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import re


# ============================================================
# 1. ACTIONS - 2 个核心操作
# ============================================================
class Action(str, Enum):
    PICK_AND_PLACE = "pick_and_place"
    OPEN = "open"

ACTION_LIST = [a.value for a in Action]


# ============================================================
# 2. OBJECT CATEGORIES - 5 个 (可被操作的物体)
# ============================================================
class ObjectCat(str, Enum):
    GRASPABLE_CONTAINER = "graspable_container"   # mug, cup, bottle, casserole, wine_bottle
    BOWL_LIKE = "bowl_like"                       # bowl, ramekin
    BOX_CONTAINER = "box_container"               # box, basket, cookie_box, caddy
    FOOD_ITEM = "food_item"                       # alphabet_soup, cream_cheese, butter, ketchup...
    ARTICULATED = "articulated"                   # drawer, cabinet_door, microwave_door

OBJECT_CAT_LIST = [c.value for c in ObjectCat]


# ============================================================
# 3. TARGET CATEGORIES - 3 个 (放置目标或被操作对象)
# ============================================================
class TargetCat(str, Enum):
    FLAT_SURFACE = "flat_surface"                 # tray, plate, cutting_board, rack, stove, table
    BOX_CONTAINER = "box_container"               # 同 object category，basket/box 也可以是 target
    ARTICULATED = "articulated"                   # drawer (open/close), cabinet_door, microwave

TARGET_CAT_LIST = [c.value for c in TargetCat]


# ============================================================
# 4. RELATIONS - 3 个核心空间关系
# ============================================================
class Relation(str, Enum):
    ON = "on"
    IN = "in"
    FROM_DRAWER = "from_drawer"

RELATION_LIST = [r.value for r in Relation]


# ============================================================
# 5. 完整数据结构
# ============================================================
@dataclass
class AbstractTask:
    operation: str = "pick_and_place"
    object_category: str = "graspable_container"
    target_category: str = "flat_surface"
    relations: List[str] = field(default_factory=list)
    num_objects: int = 1
    multi_step: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AbstractTask":
        return cls(
            operation=d.get("operation", "pick_and_place"),
            object_category=d.get("object_category", "graspable_container"),
            target_category=d.get("target_category", "flat_surface"),
            relations=d.get("relations", []),
            num_objects=d.get("num_objects", 1),
            multi_step=d.get("multi_step", False),
        )

    def to_token_sequence(self) -> List[str]:
        tokens = [f"ACT:{self.operation}"]
        tokens.append(f"OBJ:{self.object_category}")
        tokens.append(f"TGT:{self.target_category}")
        for rel in self.relations:
            tokens.append(f"REL:{rel}")
        if self.num_objects > 1:
            tokens.append(f"NUM:{self.num_objects}")
        if self.multi_step:
            tokens.append("MULTI:yes")
        return tokens

    def to_short_string(self) -> str:
        parts = [self.operation]
        if self.num_objects > 1:
            parts.append(f"×{self.num_objects}")
        parts.append(f"[{self.object_category}]")
        parts.append("→")
        parts.append(f"[{self.target_category}]")
        if self.relations:
            parts.append(f"({','.join(self.relations)})")
        if self.multi_step:
            parts.append("+multi_step")
        return " ".join(parts)


# ============================================================
# 6. 关键词映射表 (给 local parser 用)
# ============================================================
OBJECT_KEYWORDS: Dict[str, str] = {
    # graspable_container
    "mug": ObjectCat.GRASPABLE_CONTAINER.value,
    "cup": ObjectCat.GRASPABLE_CONTAINER.value,
    "wine_bottle": ObjectCat.GRASPABLE_CONTAINER.value,
    "wine bottle": ObjectCat.GRASPABLE_CONTAINER.value,
    "casserole": ObjectCat.GRASPABLE_CONTAINER.value,
    "bottle": ObjectCat.GRASPABLE_CONTAINER.value,

    # bowl_like
    "bowl": ObjectCat.BOWL_LIKE.value,
    "ramekin": ObjectCat.BOWL_LIKE.value,

    # box_container
    "basket": ObjectCat.BOX_CONTAINER.value,
    "box": ObjectCat.BOX_CONTAINER.value,
    "cookie_box": ObjectCat.BOX_CONTAINER.value,
    "cookie box": ObjectCat.BOX_CONTAINER.value,
    "caddy": ObjectCat.BOX_CONTAINER.value,
    "container": ObjectCat.BOX_CONTAINER.value,

    # food_item
    "alphabet_soup": ObjectCat.FOOD_ITEM.value,
    "alphabet soup": ObjectCat.FOOD_ITEM.value,
    "cream_cheese": ObjectCat.FOOD_ITEM.value,
    "cream cheese": ObjectCat.FOOD_ITEM.value,
    "chocolate_pudding": ObjectCat.FOOD_ITEM.value,
    "chocolate pudding": ObjectCat.FOOD_ITEM.value,
    "butter": ObjectCat.FOOD_ITEM.value,
    "ketchup": ObjectCat.FOOD_ITEM.value,
    "bbq_sauce": ObjectCat.FOOD_ITEM.value,
    "bbq sauce": ObjectCat.FOOD_ITEM.value,
    "salad_dressing": ObjectCat.FOOD_ITEM.value,
    "salad dressing": ObjectCat.FOOD_ITEM.value,
    "tomato_sauce": ObjectCat.FOOD_ITEM.value,
    "tomato sauce": ObjectCat.FOOD_ITEM.value,
    "milk": ObjectCat.FOOD_ITEM.value,
    "orange_juice": ObjectCat.FOOD_ITEM.value,
    "orange juice": ObjectCat.FOOD_ITEM.value,

    # articulated
    "drawer": ObjectCat.ARTICULATED.value,
    "cabinet": ObjectCat.ARTICULATED.value,
    "microwave": ObjectCat.ARTICULATED.value,
    "door": ObjectCat.ARTICULATED.value,
}

TARGET_KEYWORDS: Dict[str, str] = {
    # flat_surface
    "tray": TargetCat.FLAT_SURFACE.value,
    "plate": TargetCat.FLAT_SURFACE.value,
    "cutting_board": TargetCat.FLAT_SURFACE.value,
    "cutting board": TargetCat.FLAT_SURFACE.value,
    "rack": TargetCat.FLAT_SURFACE.value,
    "stove": TargetCat.FLAT_SURFACE.value,
    "table": TargetCat.FLAT_SURFACE.value,
    "surface": TargetCat.FLAT_SURFACE.value,

    # box_container (target 视角)
    "basket": TargetCat.BOX_CONTAINER.value,
    "box": TargetCat.BOX_CONTAINER.value,
    "cookie box": TargetCat.BOX_CONTAINER.value,
    "container": TargetCat.BOX_CONTAINER.value,

    # articulated (target 视角)
    "drawer": TargetCat.ARTICULATED.value,
    "cabinet": TargetCat.ARTICULATED.value,
    "microwave": TargetCat.ARTICULATED.value,
    "door": TargetCat.ARTICULATED.value,
}


def _match_in_region(region: str, kw_map: Dict[str, str]) -> Optional[str]:
    """在指定文本区域内找所有匹配，返回优先级最高的 category"""
    region = region.lower().replace("_", " ")
    found: List[tuple[int, str, str]] = []  # (位置, keyword, category)
    for kw, cat in kw_map.items():
        idx = region.find(kw)
        if idx >= 0:
            found.append((idx, kw, cat))
    if not found:
        return None
    # 优先级长度排序：更长的 keyword 更具体，优先匹配
    found.sort(key=lambda x: -len(x[1]))
    return found[0][2]


# ============================================================
# 7. 最简 parser (v2 - 角色感知)
# ============================================================
def parse_instruction(instruction: str) -> AbstractTask:
    """
    极简 parser v2: 用 regex 定位 object-region 和 target-region,
    避免 "basket" 同时匹配 object 和 target 的歧义。

    关键修复: 旧版在全文本里找第一个匹配就 break, 导致
      "pick up the alphabet_soup and place in the basket"
      里 basket (box_container) 排在 alphabet_soup (food_item) 前面,
      object 被误判成 box_container。
    """
    text = instruction.lower().strip().replace("_", " ")

    # ---- Action ----
    if re.search(r"\bopen\b", text):
        action = Action.OPEN.value
    elif re.search(r"\bturn on\b|\bpush\b", text):
        action = Action.OPEN.value if "drawer" in text else Action.PICK_AND_PLACE.value
    else:
        action = Action.PICK_AND_PLACE.value

    # ---- 定位 Object region: pick up / pick the / pick up the / move ----
    object_region = ""
    m = re.search(r"pick\s+(?:up\s+)?(?:the\s+)?(.+?)(?:\s+and\s+|\s+to\s+|\s+in\s+|\s+on\s+|$)", text)
    if m:
        object_region = m.group(1)
    else:
        m = re.search(r"move\s+(?:the\s+)?(.+?)(?:\s+to\s+|\s+on\s+|\s+in\s+|$)", text)
        if m:
            object_region = m.group(1)
        else:
            object_region = text  # fallback

    # ---- 定位 Target region: place in/on/to / put in/on/to ----
    target_region = ""
    m = re.search(r"(?:place|put)\s+(?:it\s+)?(?:in|on|to|inside)\s+(?:the\s+)?(.+?)(?:\s+and\s+|\s+inside\s+|$)", text)
    if m:
        target_region = m.group(1)
    else:
        m = re.search(r"(?:place|put)\s+(?:it\s+)?(.+?)$", text)
        if m:
            target_region = m.group(1)
        else:
            # open drawer / turn on stove 的 target 就是被操作对象本身
            if "drawer" in text or "cabinet" in text or "microwave" in text:
                target_region = text
            else:
                target_region = text

    # ---- Articulated target 的特殊处理 (open 任务) ----
    if action == Action.OPEN.value:
        object_cat = ObjectCat.ARTICULATED.value
        target_cat = TargetCat.ARTICULATED.value
    else:
        object_cat = _match_in_region(object_region, OBJECT_KEYWORDS) or ObjectCat.GRASPABLE_CONTAINER.value
        target_cat = _match_in_region(target_region, TARGET_KEYWORDS) or TargetCat.FLAT_SURFACE.value

    # ---- Relations ----
    relations: List[str] = []
    if re.search(r"\bon(?:to)?\b", text):
        relations.append(Relation.ON.value)
    if re.search(r"\bin(?:side)?\b", text):
        relations.append(Relation.IN.value)

    if re.search(r"from\s+(?:table center|the\s+)?drawer", text):
        relations.append(Relation.FROM_DRAWER.value)
    elif re.search(r"from_top_drawer|top drawer", text):
        relations.append(Relation.FROM_DRAWER.value)
    elif re.search(r"from_middle_drawer|middle drawer", text):
        relations.append(Relation.FROM_DRAWER.value)
    elif re.search(r"from_bottom_drawer|bottom drawer", text):
        relations.append(Relation.FROM_DRAWER.value)
    elif re.search(r"from\s+drawer", text):
        relations.append(Relation.FROM_DRAWER.value)

    # ---- Object count ----
    num_objects = 1
    if re.search(r"\bboth\b", text):
        num_objects = 2
    elif re.search(r"\btwo\b|\b2\b", text) and re.search(r"\b(?:moka_pot|moka pot|pot|bowl)\b", text):
        num_objects = 2

    # ---- Multi-step detection: "X and Y" where both are actions ----
    multi_step = False
    if re.search(r"\b(?:turn on|open|close|push|pull)\b.*\band\b.*\b(?:pick|place|put|move)\b", text):
        multi_step = True
    elif re.search(r"\b(?:pick|place|put|move)\b.*\band\b.*\b(?:turn on|open|close|push|pull)\b", text):
        multi_step = True

    return AbstractTask(
        operation=action,
        object_category=object_cat,
        target_category=target_cat,
        relations=relations,
        num_objects=num_objects,
        multi_step=multi_step,
    )


# ============================================================
# 8. Tokenizer (MVP)
# ============================================================
class AbstractTaskTokenizer:
    PAD_TOKEN = "[PAD]"
    CLS_TOKEN = "[CLS]"

    def __init__(self):
        self.vocab = {self.PAD_TOKEN: 0, self.CLS_TOKEN: 1}
        idx = 2
        for a in ACTION_LIST:
            self.vocab[f"ACT:{a}"] = idx; idx += 1
        for o in OBJECT_CAT_LIST:
            self.vocab[f"OBJ:{o}"] = idx; idx += 1
        for t in TARGET_CAT_LIST:
            self.vocab[f"TGT:{t}"] = idx; idx += 1
        for r in RELATION_LIST:
            self.vocab[f"REL:{r}"] = idx; idx += 1
        for n in [2, 3]:
            self.vocab[f"NUM:{n}"] = idx; idx += 1
        self.vocab["MULTI:yes"] = idx; idx += 1

        self.pad_id = self.vocab[self.PAD_TOKEN]
        self.cls_id = self.vocab[self.CLS_TOKEN]
        self.vocab_size = len(self.vocab)
        self.id2token = {v: k for k, v in self.vocab.items()}

    def tokenize(self, task: AbstractTask) -> List[int]:
        return [self.cls_id] + [self.vocab[t] for t in task.to_token_sequence()]

    def tokenize_batch(self, tasks: List[AbstractTask], max_len: int = 16) -> "torch.LongTensor":
        import torch
        all_ids = [self.tokenize(t) for t in tasks]
        padded = torch.full((len(tasks), max_len), self.pad_id, dtype=torch.long)
        for i, ids in enumerate(all_ids):
            l = min(len(ids), max_len)
            padded[i, :l] = torch.tensor(ids[:l], dtype=torch.long)
        return padded

    def detokenize(self, ids) -> List[str]:
        return [self.id2token.get(int(i), f"[{int(i)}]") for i in ids]

    def print_vocab(self):
        print(f"Tokenizer vocab ({self.vocab_size} tokens):")
        for k, v in sorted(self.vocab.items(), key=lambda x: x[1]):
            print(f"  {v:2d}: {k}")


# ============================================================
# 9. Self-test
# ============================================================
if __name__ == "__main__":
    import torch

    tokenizer = AbstractTaskTokenizer()
    tokenizer.print_vocab()

    tests = [
        # 原始文本 → MVP 抽象表示
        "Pick up the red mug and place it on the tray",
        "Pick up the blue mug and place it on the tray",
        "Pick up the black bowl and place it on the plate",
        "Put the cream cheese in the bowl",
        "Pick up the alphabet soup and place it in the basket",
        "Open the middle drawer of the cabinet",
        "Open the top drawer",
        "Turn on the stove",
        "Pick up the wine bottle and place it on the rack",
        "Pick up the black bowl from the top drawer and place it on the plate",
        "Move the ketchup bottle to the tray",
    ]

    print(f"\n{'Instruction':<55s} → {'AbstractTask':<65s} → IDs")
    print("-" * 130)
    all_ids = []
    for instr in tests:
        task = parse_instruction(instr)
        ids = tokenizer.tokenize(task)
        all_ids.append(tuple(ids))
        print(f"  {instr[:52]:<52s} → {task.to_short_string():<63s} → {ids}")

    unique = len(set(all_ids))
    print(f"\n{len(tests)} 条指令 → {unique} 个 unique token sequences "
          f"({100*unique/len(tests):.0f}% 压缩率)")

    # Encoder 验证
    from abstract_task_encoder import AbstractTaskEncoder
    encoder = AbstractTaskEncoder(
        hidden_size=64, max_len=16, num_layers=1, num_heads=4,
        vocab_size=tokenizer.vocab_size,
    )
    batch_ids = tokenizer.tokenize_batch([parse_instruction(t) for t in tests], 16)
    out = encoder(batch_ids)
    print(f"\nEncoder output: task_embedding={out['task_embedding'].shape}, "
          f"task_features={out['task_features'].shape}")

    # 余弦相似度: 相同 abstract 表示 → embedding 应该相同
    emb = out["task_embedding"]
    cos = torch.nn.functional.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
    print(f"Cosine similarity matrix (diagonal=1.0, off-diagonal 看语义接近度):")
    for i in range(len(tests)):
        print(f"  {i}: {cos[i].tolist()}")