#!/usr/bin/env python3
"""
VLA-Arena 全套件评估脚本。

对比 SimVLA 在原版 LIBERO OOD vs VLA-Arena 11 个新套件上的泛化性能。
支持 baseline / semantic / visual 三种模式。

用法:
  python eval_vla_arena.py --mode baseline --num_trials 5 --max_suites 3
  python eval_vla_arena.py --mode semantic  --num_trials 5
  python eval_vla_arena.py --mode visual    --num_trials 3
  python eval_vla_arena.py --suites safety_static_obstacles,safety_hazard_avoidance --num_trials 10
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

from register_vla_arena import register_all_vla_arena_packages
register_all_vla_arena_packages()

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

CHECKPOINT = "checkpoints/SimVLA-LIBERO"
SEMANTIC_CKPT = "checkpoints/SimVLA-LIBERO/semantic_encoder.pt"
NORM_STATS = "norm_stats/libero_norm.json"
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


def load_model(mode: str = "baseline"):
    print(f"Loading SimVLA from {CHECKPOINT} ...")
    config = SmolVLMVLAConfig.from_pretrained(CHECKPOINT)

    use_semantic = (mode == "semantic" or mode == "visual")
    use_visual = (mode == "visual")

    owl_pipe = None
    if use_visual:
        config.use_visual_semantic = True
        config.visual_semantic_hidden_size = 256
        print(f"   Visual mode enabled → loading OWL-ViT ...")
        owl_pipe = load_owl_vit_pipeline(device=device)

    if use_semantic:
        config.use_semantic = True
        config.semantic_hidden_size = 256
        config.semantic_num_layers = 2
        config.semantic_num_heads = 4
        config.semantic_max_len = 16
        tokenizer = AbstractTaskTokenizer()
        config.semantic_vocab_size = tokenizer.vocab_size
        print(f"   Semantic mode enabled: vocab={tokenizer.vocab_size}, hidden={config.semantic_hidden_size}")
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


IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def preprocess_images(img0, img1):
    img0_t = IMG_TRANSFORM(Image.fromarray(img0.astype(np.uint8)))
    img1_t = IMG_TRANSFORM(Image.fromarray(img1.astype(np.uint8)))
    padding = torch.zeros_like(img0_t)
    images = torch.stack([img0_t, img1_t, padding], dim=0)
    image_mask = torch.tensor([[True, True, False]])
    return images.unsqueeze(0).to(device), image_mask.to(device)


@torch.no_grad()
def infer(model, processor, img0, img1, state, instruction, tokenizer=None, cached_vs=None, step_idx=0, total_steps=800):
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


def quat2axisangle(q):
    if q[3] > 1.0: q[3] = 1.0
    elif q[3] < -1.0: q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def rollout_one(model, processor, suite, task_idx, num_trials, seed, tokenizer=None, owl_pipe=None):
    task = suite.get_task(task_idx)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = suite.get_task_init_states(task_idx)
    instruction = task.language

    task_success = 0
    for ep in range(num_trials):
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(seed + ep)
        obs = env.reset()
        obs = env.set_init_state(init_states[ep % len(init_states)])

        cached_vs = None
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

                if owl_pipe is not None and model.use_visual_semantic and (cached_vs is None or t % DET_EVERY == 0):
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

                actions = infer(model, processor, img, wrist_img, state, instruction,
                                tokenizer, cached_vs=cached_vs,
                                step_idx=t, total_steps=ACTUAL_MAX_STEPS)
                action = actions[0]
                obs, reward, done, info = env.step(action.tolist())

                if done:
                    task_success += 1
                    break
            except Exception as e:
                print(f"\n    ⚠️ Rollout error: {e}")
                traceback.print_exc()
                break

        env.close()

    return task_success, num_trials


def collect_vla_arena_suites():
    all_suites = benchmark.get_benchmark_dict()
    va = {}
    for k, v in sorted(all_suites.items()):
        if k.startswith("vla_arena_"):
            try:
                suite = v()
                _ = suite.tasks
                short_name = k[len("vla_arena_"):]
                va[short_name] = (k, suite)
            except Exception as e:
                print(f"⚠️ Skip {k}: {e}")
    return va


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "semantic", "visual"])
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_suites", type=int, default=0,
                        help="Limit number of suites (0=all). Useful for smoke test.")
    parser.add_argument("--suites", type=str, default=None,
                        help="Comma-separated suite short names, e.g. safety_static_obstacles,long_horizon")
    parser.add_argument("--max_tasks_per_suite", type=int, default=0,
                        help="Limit tasks per suite (0=all).")
    args = parser.parse_args()

    model, processor, tokenizer, owl_pipe = load_model(mode=args.mode)

    if model.use_visual_semantic and owl_pipe is not None:
        mode_label = "VISUAL+SEMANTIC"
    elif tokenizer is not None and model.use_semantic:
        mode_label = "SEMANTIC"
    else:
        mode_label = "BASELINE"

    all_va = collect_vla_arena_suites()
    print(f"\n发现 {len(all_va)} 个 VLA-Arena 套件")

    if args.suites:
        wanted = set(s.strip() for s in args.suites.split(","))
        all_va = {k: v for k, v in all_va.items() if k in wanted}
        print(f"  过滤后: {len(all_va)} 个套件 → {list(all_va.keys())}")

    if args.max_suites > 0:
        keys = sorted(all_va.keys())[:args.max_suites]
        all_va = {k: all_va[k] for k in keys}
        print(f"  max_suites={args.max_suites} → {len(all_va)} 个套件")

    total_tasks = sum(v[1].n_tasks for v in all_va.values())
    total_evals = total_tasks * args.num_trials
    print(f"\n{'='*70}")
    print(f"  [{mode_label}] VLA-Arena 评估")
    print(f"  {len(all_va)} 套件 × {total_tasks} 任务 × {args.num_trials} trials = {total_evals} rollouts")
    print(f"{'='*70}")

    np.random.seed(args.seed)
    overall_success, overall_episode = 0, 0
    suite_results = {}

    for suite_short_name, (suite_key, suite) in all_va.items():
        n = suite.n_tasks
        if args.max_tasks_per_suite > 0:
            n = min(n, args.max_tasks_per_suite)

        print(f"\n--- {suite_short_name} ({suite_key}) [{n}/{suite.n_tasks} tasks] ---")
        suite_success, suite_episode = 0, 0
        per_task = []

        for ti in range(n):
            task = suite.get_task(ti)
            name = task.name
            t0 = time.time()
            try:
                success, total = rollout_one(
                    model, processor, suite, ti, args.num_trials, args.seed,
                    tokenizer=tokenizer, owl_pipe=owl_pipe,
                )
            except Exception as e:
                print(f"  ⚠️  [{ti:2d}/{n}] EXC: {type(e).__name__}: {str(e)[:80]}")
                success, total = 0, args.num_trials
            dt = time.time() - t0
            rate = success / total * 100
            suite_success += success
            suite_episode += total
            overall_success += success
            overall_episode += total
            per_task.append({"task": name, "language": task.language, "successes": success, "trials": total, "rate": rate})
            marker = "✅" if success > 0 else "❌"
            print(f"  {marker} [{ti:2d}/{n}] {rate:5.1f}% ({success}/{total})  {name[:50]:50s}  {dt:.0f}s")

        suite_rate = suite_success / max(suite_episode, 1) * 100
        suite_results[suite_short_name] = {
            "suite_key": suite_key,
            "n_tasks": n,
            "successes": suite_success,
            "episodes": suite_episode,
            "rate": suite_rate,
            "per_task": per_task,
        }
        print(f"  Suite OVERALL: {suite_rate:.1f}% ({suite_success}/{suite_episode})")

    overall_rate = overall_success / max(overall_episode, 1) * 100

    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"vla_arena_{args.mode}_t{args.num_trials}.json"
    with open(out_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "mode_label": mode_label,
            "num_trials": args.num_trials,
            "seed": args.seed,
            "overall_rate": overall_rate,
            "total_success": overall_success,
            "total_episodes": overall_episode,
            "suites": suite_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY [{mode_label}]")
    print(f"{'='*70}")
    print(f"  {'Suite':<45s} {'Tasks':>5s} {'Success':>10s} {'Rate':>8s}")
    print(f"  {'-'*68}")
    for sn, sr in suite_results.items():
        bar = "█" * int(sr["rate"] / 5)
        print(f"  {sn:<45s} {sr['n_tasks']:5d} {sr['successes']:>5d}/{sr['episodes']:<5d} {sr['rate']:>6.1f}%  {bar}")
    print(f"  {'-'*68}")
    print(f"  {'OVERALL':<45s} {sum(s['n_tasks'] for s in suite_results.values()):5d} {overall_success:>5d}/{overall_episode:<5d} {overall_rate:>6.1f}%")
    print(f"{'='*70}")
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()