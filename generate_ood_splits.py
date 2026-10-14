"""
Generate 4 principled OOD splits from LIBERO 40-task dataset.

严格定义 (semantic abstraction 视角):

  ID (In-Distribution):
    覆盖 5 种 object category + 3 种 target category。
    训练目标: 让模型见过所有 category-level affordance。

  OOD-Instance:
    - (operation, object_cat, target_cat) 三元组 ∈ ID
    - 但 object 的具体 instance (alphabet_soup vs cream_cheese) ∉ ID
    - 测试: 去实例化能否泛化到同一 category 下的新物体

  OOD-Composition:
    - object_cat ∈ ID, target_cat ∈ ID (category level 都见过)
    - 但 (operation, object_cat, target_cat) 这个**组合** ∉ ID
    - 测试: 见过 bowl_like 和见过 flat_surface, 能否 bowl_like → box_container?

  OOD-Both:
    - object_cat ∉ ID 或 target_cat ∉ ID
    - 或多步复合任务 (LIBERO 10 的特色)
    - 测试: 最严格 OOD — category 和 composition 都是新的

Usage: python generate_ood_splits.py
"""

from __future__ import annotations
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

from models.semantic_schema import parse_instruction, AbstractTask

# ============================================================
PATHS = {
    "libero_object": "datasets/metas/libero_object",
    "libero_goal":    "datasets/metas/libero_goal",
    "libero_spatial": "datasets/metas/libero_spatial",
    "libero_10":      "datasets/metas/libero_10",
}
OUT_DIR = Path("datasets/ood_splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
def scan_all() -> List[Dict]:
    tasks = []
    for suite, path in PATHS.items():
        for h5 in sorted(glob.glob(os.path.join(path, "*.hdf5"))):
            tn = os.path.basename(h5).replace("_demo.hdf5", "")
            sem = parse_instruction(tn)
            tasks.append({
                "suite": suite,
                "task_name": tn,
                "hdf5_path": h5,
                "semantic": sem,
                "triple": (sem.operation, sem.object_category, sem.target_category),
            })
    return tasks


def instance_tokens(task_name: str) -> Set[str]:
    """提取任务名里的具体物体实例 token (用于判断 OOD-Instance)"""
    text = task_name.lower().replace("_", " ")
    instances = set()
    keywords = [
        "mug", "cup", "wine bottle", "wine_bottle", "casserole",
        "bowl", "ramekin",
        "basket", "cookie box", "cookie_box", "caddy",
        "alphabet soup", "alphabet_soup", "cream cheese", "cream_cheese",
        "chocolate pudding", "chocolate_pudding", "butter",
        "ketchup", "bbq sauce", "bbq_sauce", "salad dressing", "salad_dressing",
        "tomato sauce", "tomato_sauce", "milk", "orange juice", "orange_juice",
        "stove", "plate", "rack", "cutting board", "cutting_board",
        "drawer", "cabinet", "microwave", "door",
        "book", "moka pot", "moka_pot", "tray",
    ]
    for kw in keywords:
        if kw in text:
            instances.add(kw)
    return instances


# ============================================================
def is_multi_step(task_name: str) -> bool:
    """
    判断是否多步复合任务。
    精准特征:
      - 两个独立动作 (open + put / turn_on + put / put_both)
      - "both" (处理多个物体)
      - "left ... and ... right" (两个独立 placement)
    单步 pick-and-place 的 "pick_up_X_and_place_in_Y" 不算多步!
    """
    text = task_name.lower()

    # 多步明确标志: both / 双物体 / left+right
    if "put_both" in text or "both_moka" in text or "both_the" in text:
        return True
    if "left_plate_and_put" in text or "right_plate_and_put" in text:
        return True

    # 两个不同动作动词: turn_on + put / open + put / push + put
    has_turn_on = "turn_on" in text or "turn on" in text
    has_open = ("open_" in text or "_open_" in text or " open " in text) and "drawer" in text
    has_put_after = "_and_put_" in text or " and put " in text or "and_close_it" in text

    if has_turn_on and has_put_after:
        return True
    if has_open and has_put_after:
        return True

    # pick_up_the_book_and_place_in_caddy —— 虽然是一个 pick+place, 但 caddy 有 back compartment, 是复合 placement
    if "back_compartment" in text:
        return True

    return False


def build_splits(all_tasks: List[Dict]) -> Dict[str, List[Dict]]:

    # ---- Step 1: ID triple 覆盖 ----
    # 覆盖全部 5 object cat × 3 target cat × 4 种 action
    id_triples_target = {
        ("pick_and_place", "bowl_like", "flat_surface"),        # 12个
        ("pick_and_place", "food_item", "box_container"),        # 13个 (选单步)
        ("pick_and_place", "graspable_container", "flat_surface"), # 7个
        ("open", "articulated", "articulated"),                  # 2个
    }

    id_tasks: List[Dict] = []
    ood_candidates: List[Dict] = []

    for t in all_tasks:
        triple = t["triple"]
        if triple in id_triples_target:
            id_tasks.append(t)
        else:
            ood_candidates.append(t)

    # food_item→box_container (13个): 留 3 个单步在 ID, 其余 (10个单步+3个多步) → OOD-candidates
    # 目的: OOD-Instance 里有大量同 triple 不同实例的任务
    food_box = [t for t in id_tasks if t["triple"] == ("pick_and_place", "food_item", "box_container")]
    if len(food_box) > 3:
        keep = food_box[:3]
        move = food_box[3:]
        id_tasks = [t for t in id_tasks if t not in move]
        ood_candidates.extend(move)

    # graspable_container→flat_surface (7个): 留 3 个在 ID, 其余 → OOD-candidates (做 OOD-Instance)
    grasp_flat = [t for t in id_tasks if t["triple"] == ("pick_and_place", "graspable_container", "flat_surface")]
    if len(grasp_flat) > 3:
        keep = grasp_flat[:3]
        move = grasp_flat[3:]
        id_tasks = [t for t in id_tasks if t not in move]
        ood_candidates.extend(move)

    # bowl_like→flat_surface (12个): 全部保留在 ID (spatial suite 全是这个 triple, 覆盖 spatial pattern 多样性)
    # articulated→articulated (2个): 全部保留

    # ---- Step 2: OOD-candidates 分桶 ----
    id_triples: Set[Tuple] = {t["triple"] for t in id_tasks}
    id_obj_cats: Set[str] = {t["semantic"].object_category for t in id_tasks}
    id_tgt_cats: Set[str] = {t["semantic"].target_category for t in id_tasks}
    id_multistep = {is_multi_step(t["task_name"]) for t in id_tasks}

    ood_instance = []
    ood_composition = []
    ood_both = []

    for t in ood_candidates:
        triple = t["triple"]
        obj_cat = t["semantic"].object_category
        tgt_cat = t["semantic"].target_category
        multistep = is_multi_step(t["task_name"])

        # 多步任务 → OOD-Both (即使 triple 匹配, 多步本身就是 OOD)
        if multistep:
            ood_both.append(t)
        elif triple in id_triples:
            # triple 相同 + 单步 → OOD-Instance (纯实例级 OOD)
            ood_instance.append(t)
        elif obj_cat in id_obj_cats and tgt_cat in id_tgt_cats:
            # category 见过 + triple 新 + 单步 → OOD-Composition
            ood_composition.append(t)
        else:
            # category 没见过 → 极端 OOD-Both
            ood_both.append(t)

    return {
        "ID": id_tasks,
        "OOD-Instance": ood_instance,
        "OOD-Composition": ood_composition,
        "OOD-Both": ood_both,
    }


# ============================================================
def print_stats(splits: Dict[str, List[Dict]]):
    print("\n" + "=" * 80)
    print("SPLIT STATISTICS (Principled OOD)")
    print("=" * 80)

    # 全局 ID category coverage
    id_obj = {t["semantic"].object_category for t in splits["ID"]}
    id_tgt = {t["semantic"].target_category for t in splits["ID"]}
    id_triples = {t["triple"] for t in splits["ID"]}
    all_obj = {t["semantic"].object_category for t in sum(splits.values(), [])}
    all_tgt = {t["semantic"].target_category for t in sum(splits.values(), [])}

    print(f"\nID coverage:")
    print(f"  Object categories: {sorted(id_obj)}  (full: {sorted(all_obj)})")
    print(f"  Target categories: {sorted(id_tgt)}  (full: {sorted(all_tgt)})")
    print(f"  Semantic triples ({len(id_triples)}):")
    for tr in sorted(id_triples):
        print(f"    {tr}")

    for split_name, tasks in splits.items():
        print(f"\n【{split_name}】({len(tasks)} tasks)")
        # suite 分布
        suite_counts = defaultdict(int)
        for t in tasks:
            suite_counts[t["suite"]] += 1
        for s, c in sorted(suite_counts.items()):
            print(f"    {s}: {c}")

        # triple 分布
        triple_counts = defaultdict(int)
        for t in tasks:
            triple_counts[t["triple"]] += 1
        for tr, c in sorted(triple_counts.items()):
            marker = "∈ ID" if tr in id_triples else "∉ ID"
            print(f"    [{c:2d}x] {tr[0]:20s} | {tr[1]:22s} | {tr[2]:20s}  ({marker})")

        # 列出任务名
        for t in tasks:
            print(f"      - {t['task_name']}")

    # 无 overlap 验证
    print(f"\n{'='*80}")
    all_names = {name: {t["task_name"] for t in tasks} for name, tasks in splits.items()}
    ok = True
    names_list = list(all_names.items())
    for i, (n1, s1) in enumerate(names_list):
        for n2, s2 in names_list[i+1:]:
            overlap = s1 & s2
            if overlap:
                print(f"  ⚠️  OVERLAP {n1} ∩ {n2}: {overlap}")
                ok = False
    if ok:
        print("  ✅ All 4 splits are mutually exclusive")

    total = sum(len(t) for t in splits.values())
    print(f"\n  Total: {total} tasks")


# ============================================================
def save_splits(splits: Dict[str, List[Dict]]):
    for split_name, tasks in splits.items():
        out = []
        for t in tasks:
            out.append({
                "task_name": t["task_name"],
                "suite": t["suite"],
                "hdf5_path": t["hdf5_path"],
                "semantic": t["semantic"].to_dict(),
                "triple": list(t["triple"]),
            })
        path = OUT_DIR / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  ✅ {split_name}: {len(out):2d} tasks → {path}")

    manifest = {}
    for split_name, tasks in splits.items():
        manifest[split_name] = [{
            "task_name": t["task_name"],
            "suite": t["suite"],
            "semantic": t["semantic"].to_dict(),
            "triple": list(t["triple"]),
        } for t in tasks]

    id_triples = {t["triple"] for t in splits["ID"]}
    manifest["_meta"] = {
        "id_triples": [list(tr) for tr in sorted(id_triples)],
        "id_obj_categories": sorted({t["semantic"].object_category for t in splits["ID"]}),
        "id_tgt_categories": sorted({t["semantic"].target_category for t in splits["ID"]}),
    }

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  ✅ manifest.json")


# ============================================================
if __name__ == "__main__":
    print("Scanning LIBERO 40-task dataset...")
    all_tasks = scan_all()
    print(f"  Total: {len(all_tasks)} tasks")

    print("\nBuilding principled OOD splits...")
    splits = build_splits(all_tasks)

    print_stats(splits)
    print()
    save_splits(splits)

    print(f"\n{'='*80}")
    print("Split design (semantic abstraction perspective):")
    print("-" * 80)
    print("  ID:              训练集, 覆盖 5 object cat × 3 target cat")
    print("  OOD-Instance:     (op, obj_cat, tgt_cat) ∈ ID, 但具体物体实例 ∉ ID")
    print("  OOD-Composition:  obj_cat ∈ ID, tgt_cat ∈ ID, 但三元组 ∉ ID")
    print("  OOD-Both:         obj_cat ∉ ID 或 tgt_cat ∉ ID, 或多步复合任务")
    print("=" * 80)