#!/usr/bin/env python3
"""
LIBERO-PRO Language Perturbation Eval.
策略: 生成两份 split JSON (orig / perturbed)，直接复用 eval_ood_splits.py 的 eval_split。

用法:
  python eval_libero_pro.py --mode baseline --num_trials 10
  python eval_libero_pro.py --mode semantic --num_trials 10
  python eval_libero_pro.py --mode visual   --num_trials 10
"""
from __future__ import annotations
import argparse, json, os, sys, yaml
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "libero" / "LIBERO"))

from eval_ood_splits import load_model, eval_split

LANG_CFG = "datasets/ood_language.yaml"
WORK_DIR = "datasets/ood_pro_splits"

def build_splits(lang_cfg: dict, perturb_seed: int = 42):
    """从 ood_language.yaml 生成 orig 和 pert 两份 split JSON"""
    rng = np.random.RandomState(perturb_seed)
    orig_tasks, pert_tasks = [], []
    for suite_name, task_dict in lang_cfg.items():
        for task_name, candidates in task_dict.items():
            orig_tasks.append({"task_name": task_name, "suite": suite_name})
            str_cands = [c for c in candidates if isinstance(c, str)]
            if not str_cands:
                print(f'  ⚠️ Skip {task_name}: no string candidates in {candidates}')
                continue
            chosen = rng.choice(str_cands)
            pert_tasks.append({"task_name": task_name, "suite": suite_name, "instruction": chosen})
    os.makedirs(WORK_DIR, exist_ok=True)
    orig_path = os.path.join(WORK_DIR, "orig.json")
    pert_path = os.path.join(WORK_DIR, "pert.json")
    with open(orig_path, "w") as f:
        json.dump(orig_tasks, f, indent=2)
    with open(pert_path, "w") as f:
        json.dump(pert_tasks, f, indent=2)
    print(f"Splits: {len(orig_tasks)} tasks each -> {orig_path}, {pert_path}")
    return orig_path, pert_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="semantic", choices=["baseline", "semantic", "visual"])
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--perturb_seed", type=int, default=42)
    args = parser.parse_args()

    with open(LANG_CFG) as f:
        lang_cfg = yaml.safe_load(f) or {}

    model, processor, tokenizer, owl_pipe = load_model(args.mode)
    orig_path, pert_path = build_splits(lang_cfg, args.perturb_seed)

    print("\n" + "=" * 60)
    print("Phase 1: ORIGINAL instruction")
    print("=" * 60)
    eval_split(model, processor, orig_path, args.num_trials, args.seed, tokenizer, owl_pipe)

    print("\n" + "=" * 60)
    print("Phase 2: PERTURBED instruction (ood_language.yaml)")
    print("=" * 60)
    eval_split(model, processor, pert_path, args.num_trials, args.seed, tokenizer, owl_pipe)

if __name__ == "__main__":
    main()
