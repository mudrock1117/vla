#!/usr/bin/env python3
"""
直接按 OOD split 文件评估原始 SimVLA baseline。

不走 WebSocket, 直接加载模型 + 直接调 LIBERO env, 速度快。
用法:
  python eval_ood_splits.py --split ID --num_trials 10
  python eval_ood_splits.py --split OOD-Instance --num_trials 20
  python eval_ood_splits.py --split all   --num_trials 10
"""

from __future__ import annotations
import argparse, json, math, os, sys, time, traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "libero" / "LIBERO"))

from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor
from models.configuration_smolvlm_vla import SmolVLMVLAConfig
from models.semantic_schema import parse_instruction, AbstractTaskTokenizer
from models.visual_grounding import (
    VisualSemantic,
    load_owl_vit_pipeline, detect_objects, build_visual_semantic,
)

import safetensors.torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# ============================================================
# Config
# ============================================================
CHECKPOINT = "checkpoints/SimVLA-LIBERO"
SEMANTIC_CKPT = "checkpoints/SimVLA-LIBERO/semantic_encoder.pt"
NORM_STATS = "norm_stats/libero_norm.json"
SPLITS_DIR = "datasets/ood_splits"
SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
NUM_STEPS_WAIT = 10
MAX_STEPS = 800

CONFIG = {
    "image_size": 384,
    "action_horizon": 10,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Model loading
# ============================================================
def load_model(mode: str = "baseline"):
    print(f"Loading SimVLA from {CHECKPOINT} ...")
    config = SmolVLMVLAConfig.from_pretrained(CHECKPOINT)

    use_semantic = (mode == "semantic" or mode == "visual")
    use_visual = (mode == "visual")

    owl_pipe = None
    if use_visual:
        config.use_visual_semantic = True
        config.visual_semantic_hidden_size = 256
        print(f"   Visual mode enabled")
        print(f"   Loading OWL-ViT for visual grounding ...")
        owl_pipe = load_owl_vit_pipeline(device=device)

    if use_semantic:
        config.use_semantic = True
        config.semantic_hidden_size = 256
        config.semantic_num_layers = 2
        config.semantic_num_heads = 4
        config.semantic_max_len = 16
        tokenizer = AbstractTaskTokenizer()
        config.semantic_vocab_size = tokenizer.vocab_size
        print(f"   Semantic mode enabled: vocab={tokenizer.vocab_size}, "
              f"hidden={config.semantic_hidden_size}")
    else:
        tokenizer = None

    model = SmolVLMVLA(config)
    state = safetensors.torch.load_file(os.path.join(CHECKPOINT, "model.safetensors"))

    sem_state = {}
    if os.path.exists(SEMANTIC_CKPT):
        sem_state = torch.load(SEMANTIC_CKPT, map_location="cpu")

    if sem_state:
        state.update(sem_state)
        print(f"   Loaded semantic+visual encoder weights ({len(sem_state)} keys)")

    missing, unexpected = model.load_state_dict(state, strict=False)

    sem_missing = [k for k in missing if "semantic" in k or "visual" in k]
    other_missing = [k for k in missing if "semantic" not in k and "visual" not in k]
    if sem_missing:
        print(f"   ⚠️ Semantic/Visual weights NOT found ({len(sem_missing)} keys) — using random init")
    if other_missing:
        print(f"   ⚠️ Other missing: {len(other_missing)} keys")
    print(f"   unexpected={len(unexpected)}")

    if use_semantic:
        print(f"   α_sem={torch.sigmoid(model.semantic_alpha).item():.4f}")
    if use_visual:
        print(f"   α_vis={torch.sigmoid(model.visual_alpha).item():.4f}")

    model = model.to(device).eval()

    processor = SmolVLMVLAProcessor.from_pretrained(SMOLVLM_MODEL)

    if os.path.exists(NORM_STATS):
        model.action_space.load_norm_stats(NORM_STATS)
        print(f"   Norm stats loaded from {NORM_STATS}")
    else:
        print(f"   WARNING: No norm stats at {NORM_STATS}")

    print(f"   Device: {device}")
    return model, processor, tokenizer, owl_pipe


# ============================================================
# Inference
# ============================================================
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def preprocess_images(img0: np.ndarray, img1: np.ndarray):
    img0_t = IMG_TRANSFORM(Image.fromarray(img0.astype(np.uint8)))
    img1_t = IMG_TRANSFORM(Image.fromarray(img1.astype(np.uint8)))
    padding = torch.zeros_like(img0_t)
    images = torch.stack([img0_t, img1_t, padding], dim=0)
    image_mask = torch.tensor([[True, True, False]])
    return images.unsqueeze(0).to(device), image_mask.to(device)


@torch.no_grad()
def infer(model, processor, img0, img1, state, instruction, tokenizer=None, owl_pipe=None, cached_vs=None, step_idx=0, total_steps=800):
    images, image_mask = preprocess_images(img0, img1)
    proprio = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    lang = processor.encode_language([instruction])
    lang = {k: v.to(device) for k, v in lang.items()}

    abstract_task_ids = None
    visual_semantics = None

    if tokenizer is not None and model.use_semantic:
        task = parse_instruction(instruction)
        abstract_task_ids = tokenizer.tokenize_batch([task], 16).to(device)

    if cached_vs is not None:
        visual_semantics = [cached_vs]
    elif owl_pipe is not None and model.use_visual_semantic:
        try:
            dets = detect_objects(owl_pipe, img0, confidence_threshold=0.05)
            vs = build_visual_semantic(dets, img0.shape[0], img0.shape[1], step_idx=step_idx, total_steps=total_steps)
            visual_semantics = [vs]
        except Exception:
            visual_semantics = [VisualSemantic()]

    actions = model.generate_actions(
        input_ids=lang["input_ids"],
        image_input=images,
        image_mask=image_mask,
        proprio=proprio,
        steps=CONFIG["action_horizon"],
        abstract_task_ids=abstract_task_ids,
        visual_semantics=visual_semantics,
    )
    return actions.cpu().numpy()[0]


# ============================================================
# Utilities
# ============================================================
def quat2axisangle(q: np.ndarray) -> np.ndarray:
    if q[3] > 1.0: q[3] = 1.0
    elif q[3] < -1.0: q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def get_libero_task(task_name: str):
    """从 LIBERO benchmark 里按 task_name 找任务"""
    benchmark_dict = benchmark.get_benchmark_dict()

    for suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        suite = benchmark_dict[suite_name]()
        for i in range(suite.n_tasks):
            task = suite.get_task(i)
            if task.name == task_name or task.language == task_name:
                return task, suite, i
    return None, None, None


# ============================================================
# Eval one split
# ============================================================
def eval_split(model, processor, split_path: str, num_trials: int = 10, seed: int = 7, tokenizer=None, owl_pipe=None):
    with open(split_path) as f:
        tasks = json.load(f)

    split_name = Path(split_path).stem
    if model.use_visual_semantic and owl_pipe is not None:
        mode_label = "VISUAL+SEMANTIC"
    elif tokenizer is not None and model.use_semantic:
        mode_label = "SEMANTIC"
    else:
        mode_label = "BASELINE"
    print(f"\n{'='*70}")
    print(f"  [{mode_label}] SPLIT: {split_name}  ({len(tasks)} tasks × {num_trials} trials)")
    print(f"{'='*70}")

    np.random.seed(seed)

    results = []
    total_success, total_episode = 0, 0

    for entry in tasks:
        task_name = entry["task_name"]
        suite_name = entry.get("suite", "")
        semantic = entry.get("semantic", {})

        task, suite, task_id = get_libero_task(task_name)
        if task is None:
            print(f"  ⚠️ SKIP: {task_name} not found in LIBERO benchmark")
            continue

        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_states = suite.get_task_init_states(task_id)
        instruction = entry.get("instruction", task.language)

        print(f"  Loading: {task_name[:55]:55s}", end=" ", flush=True)
        task_success = 0
        for ep in range(num_trials):
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_file),
                camera_heights=LIBERO_ENV_RESOLUTION,
                camera_widths=LIBERO_ENV_RESOLUTION,
            )
            env.seed(seed)
            obs = env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])

            ep_success = False
            cached_vs = None
            prev_vs = None
            DET_EVERY = 10
            ACTUAL_MAX_STEPS = MAX_STEPS + NUM_STEPS_WAIT
            for t in range(ACTUAL_MAX_STEPS):
                try:
                    if t < NUM_STEPS_WAIT:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    state = np.concatenate([
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ])

                    use_owl = (owl_pipe is not None and model.use_visual_semantic)
                    if use_owl and (cached_vs is None or t % DET_EVERY == 0):
                        try:
                            dets = detect_objects(owl_pipe, img, confidence_threshold=0.05)
                            prev_vs = cached_vs
                            cached_vs = build_visual_semantic(
                                dets, img.shape[0], img.shape[1],
                                step_idx=t, total_steps=ACTUAL_MAX_STEPS,
                                prev_vs=prev_vs,
                            )
                        except Exception:
                            if cached_vs is None:
                                cached_vs = VisualSemantic()

                    visual_semantics = [cached_vs] if cached_vs is not None else None

                    actions = infer(model, processor, img, wrist_img, state, instruction,
                                    tokenizer, owl_pipe=None, cached_vs=cached_vs,
                                    step_idx=t, total_steps=ACTUAL_MAX_STEPS)
                    action = actions[0]
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_success += 1
                        ep_success = True
                        break
                except Exception as e:
                    print(f"\n    ⚠️ Rollout error: {e}")
                    traceback.print_exc()
                    break

            env.close()
            total_episode += 1

        total_success += task_success
        rate = task_success / num_trials * 100
        results.append({
            "task_name": task_name,
            "suite": suite_name,
            "semantic": semantic,
            "successes": task_success,
            "trials": num_trials,
            "rate": rate,
        })
        print(f"→ {rate:5.1f}% ({task_success}/{num_trials})")

    overall = total_success / max(total_episode, 1) * 100
    print(f"\n  {split_name} OVERALL: {total_success}/{total_episode} = {overall:.1f}%")

    # Save per-split results
    out_path = f"eval_results/{split_name}.json"
    Path("eval_results").mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "split": split_name,
            "overall_rate": overall,
            "total_success": total_success,
            "total_episodes": total_episode,
            "per_task": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {out_path}")

    return overall, results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="all",
                        choices=["ID", "OOD-Instance", "OOD-Composition", "OOD-Both", "all"])
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "semantic", "visual"])
    args = parser.parse_args()

    model, processor, tokenizer, owl_pipe = load_model(mode=args.mode)

    splits_to_run = [args.split] if args.split != "all" else \
        ["ID", "OOD-Instance", "OOD-Composition", "OOD-Both"]

    all_summary = {}
    for split_name in splits_to_run:
        split_path = os.path.join(SPLITS_DIR, f"{split_name}.json")
        if not os.path.exists(split_path):
            print(f"⚠️ Split file not found: {split_path}")
            continue
        overall, _ = eval_split(
            model, processor, split_path,
            num_trials=args.num_trials, seed=args.seed,
            tokenizer=tokenizer, owl_pipe=owl_pipe,
        )
        all_summary[split_name] = overall

    # Final summary table
    if len(all_summary) > 1:
        print(f"\n{'='*70}")
        print(f"  SUMMARY [{args.mode.upper()}] ({args.num_trials} trials/task)")
        print(f"{'='*70}")
        print(f"  {'Split':<20s} {'Success Rate':>12s}")
        print(f"  {'-'*32}")
        for name, rate in all_summary.items():
            bar = "█" * int(rate / 5)
            print(f"  {name:<20s} {rate:>10.1f}%  {bar}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()