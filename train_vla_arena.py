#!/usr/bin/env python3
"""
VLA-Arena trajectory collection + fine-tune semantic/visual encoder.

复用 train_visual_semantic.py 的 model loader 和 train_step,
但 trajectory collection 遍历 VLA-Arena 套件。

用法:
  python train_vla_arena.py --suites extrapolation_preposition_combinations,long_horizon --traj_steps 40 --steps 800
  python train_vla_arena.py --all --traj_steps 30 --steps 5000
  python train_vla_arena.py --skip_owl --suites long_horizon --steps 300 --debug  # smoke test
"""

from __future__ import annotations
import argparse, json, math, os, sys, time, traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "libero" / "LIBERO"))

from register_vla_arena import register_all_vla_arena_packages
register_all_vla_arena_packages()

from models.configuration_smolvlm_vla import SmolVLMVLAConfig
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor
from models.semantic_schema import parse_instruction, AbstractTaskTokenizer
from models.visual_grounding import (
    VisualSemantic, DetectedObject,
    load_owl_vit_pipeline, detect_objects, build_visual_semantic,
)

import safetensors.torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

CHECKPOINT = "checkpoints/SimVLA-LIBERO"
NORM_STATS = "norm_stats/libero_norm.json"
SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
OUT_DIR = "checkpoints/SimVLA-VLA-Arena"

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
NUM_STEPS_WAIT = 10

device = "cuda" if torch.cuda.is_available() else "cpu"


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    if q[3] > 1.0: q[3] = 1.0
    elif q[3] < -1.0: q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def preprocess_images(img0, img1):
    img0_t = IMG_TRANSFORM(Image.fromarray(img0.astype(np.uint8)))
    img1_t = IMG_TRANSFORM(Image.fromarray(img1.astype(np.uint8)))
    padding = torch.zeros_like(img0_t)
    images = torch.stack([img0_t, img1_t, padding], dim=0)
    image_mask = torch.tensor([[True, True, False]])
    return images.unsqueeze(0).to(device), image_mask.to(device)


def _serialize_vs(vs: VisualSemantic) -> dict:
    return {
        "objects": [
            {"category": o.category, "bbox": o.bbox, "confidence": o.confidence, "label_text": o.label_text}
            for o in vs.objects
        ],
        "num_graspable_containers": vs.num_graspable_containers,
        "num_bowl_like": vs.num_bowl_like,
        "num_food_items": vs.num_food_items,
        "num_box_containers": vs.num_box_containers,
        "num_flat_surfaces": vs.num_flat_surfaces,
        "num_articulated": vs.num_articulated,
        "image_width": vs.image_width,
        "image_height": vs.image_height,
    }


def _deserialize_vs(d: dict) -> VisualSemantic:
    objs = [DetectedObject(**o) for o in d.get("objects", [])]
    return VisualSemantic(
        objects=objs,
        num_graspable_containers=d.get("num_graspable_containers", 0),
        num_bowl_like=d.get("num_bowl_like", 0),
        num_food_items=d.get("num_food_items", 0),
        num_box_containers=d.get("num_box_containers", 0),
        num_flat_surfaces=d.get("num_flat_surfaces", 0),
        num_articulated=d.get("num_articulated", 0),
        image_width=d.get("image_width", 0),
        image_height=d.get("image_height", 0),
    )


def load_models():
    print(f"Loading models from {CHECKPOINT} ...")
    tokenizer = AbstractTaskTokenizer()

    config_triple = SmolVLMVLAConfig.from_pretrained(CHECKPOINT)
    config_triple.use_semantic = True
    config_triple.use_visual_semantic = True
    config_triple.semantic_hidden_size = 256
    config_triple.semantic_num_layers = 2
    config_triple.semantic_num_heads = 4
    config_triple.semantic_max_len = 16
    config_triple.semantic_vocab_size = tokenizer.vocab_size

    model_triple = SmolVLMVLA(config_triple)
    state = safetensors.torch.load_file(os.path.join(CHECKPOINT, "model.safetensors"))
    missing, unexpected = model_triple.load_state_dict(state, strict=False)
    print(f"  Loaded baseline weights. missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"    missing keys (new modules): {missing[:5]}...")
    model_triple = model_triple.to(device)

    for name, p in model_triple.named_parameters():
        if not any(k in name for k in ["semantic", "visual"]):
            p.requires_grad = False

    trainable = sum(p.numel() for p in model_triple.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_triple.parameters())
    print(f"  Trainable: {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M")

    model_triple.action_space.load_norm_stats(NORM_STATS)

    config_base = SmolVLMVLAConfig.from_pretrained(CHECKPOINT)
    model_base = SmolVLMVLA(config_base)
    model_base.load_state_dict(state, strict=True)
    model_base = model_base.to(device).eval()
    for p in model_base.parameters():
        p.requires_grad = False
    model_base.action_space.load_norm_stats(NORM_STATS)
    print(f"  Baseline model loaded (frozen teacher)")

    processor = SmolVLMVLAProcessor.from_pretrained(SMOLVLM_MODEL)
    print(f"  Device: {device}")

    return model_base, model_triple, processor, tokenizer


def collect_trajectory_vla(model_base, owl_pipe, processor, suite, task_idx,
                            num_steps=100, det_every=5):
    task = suite.get_task(task_idx)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(7)
    init_states = suite.get_task_init_states(task_idx)
    instruction = task.language
    env.reset()
    env.set_init_state(init_states[0])

    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)

    trajectory = []
    cached_vs = None
    prev_vs = None
    with torch.no_grad():
        for t in range(num_steps):
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            state = np.concatenate([
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ])

            images, image_mask = preprocess_images(img, wrist_img)
            proprio = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            lang = processor.encode_language([instruction])
            lang = {k: v.to(device) for k, v in lang.items()}

            actions = model_base.generate_actions(
                input_ids=lang["input_ids"],
                image_input=images,
                image_mask=image_mask,
                proprio=proprio,
                steps=10,
            )

            if owl_pipe is not None and t % det_every == 0:
                try:
                    dets = detect_objects(owl_pipe, img, confidence_threshold=0.05)
                    prev_vs = cached_vs
                    cached_vs = build_visual_semantic(
                        dets, img.shape[0], img.shape[1],
                        step_idx=t, total_steps=num_steps,
                        prev_vs=prev_vs,
                    )
                except Exception:
                    cached_vs = VisualSemantic(image_width=img.shape[1], image_height=img.shape[0])

            trajectory.append({
                "images": images,
                "image_mask": image_mask,
                "proprio": proprio,
                "instruction": instruction,
                "input_ids": lang["input_ids"].cpu(),
                "abstract_ids": parse_instruction(instruction),
                "visual_semantic": _serialize_vs(cached_vs) if cached_vs is not None else None,
                "teacher_action": actions,
            })

            action = actions.cpu().numpy()[0, 0]
            obs, reward, done, info = env.step(action.tolist())
            if done:
                trajectory.append({
                    "images": images,
                    "image_mask": image_mask,
                    "proprio": proprio,
                    "instruction": instruction,
                    "input_ids": lang["input_ids"].cpu(),
                    "abstract_ids": parse_instruction(instruction),
                    "visual_semantic": _serialize_vs(cached_vs) if cached_vs is not None else None,
                    "teacher_action": torch.zeros(actions.shape),
                })
                break

    env.close()
    return trajectory


def train_step(model_triple, optimizer, batch, tokenizer):
    model_triple.train()
    optimizer.zero_grad()

    abstract_ids = tokenizer.tokenize_batch(batch["abstract_tasks"], 16).to(device)
    input_ids = batch["input_ids"].to(device)
    visual_semantics = [_deserialize_vs(d) if d is not None else VisualSemantic() for d in batch["visual_semantics"]]

    enc = model_triple.forward_vlm_semantic(
        batch["images"], batch["image_mask"], input_ids, abstract_ids, visual_semantics
    )

    teacher_action_norm = batch["teacher_action_norm"]
    B = teacher_action_norm.shape[0]
    noise = torch.randn_like(teacher_action_norm)
    t = torch.rand(B, device=device) * 0.998 + 0.001
    t_exp = t.view(-1, 1, 1)
    x_t = t_exp * noise + (1 - t_exp) * teacher_action_norm
    u_t = noise - teacher_action_norm

    v_t = model_triple.transformer(
        vlm_features=enc["vlm_features"],
        action_with_noise=x_t,
        t=t,
        proprio=batch["proprio_norm"],
    )

    loss = torch.mean(torch.square(v_t - u_t))

    aux_loss = torch.tensor(0.0, device=device)
    if model_triple.use_visual_semantic and model_triple.visual_semantic_encoder is not None:
        vis_tokens = model_triple.visual_semantic_encoder(visual_semantics)
        count_token = vis_tokens[:, 0, :]
        batch_counts = torch.tensor(
            [min(len(vs.objects), 8) for vs in visual_semantics],
            device=device, dtype=torch.float32
        )
        pred_norms = count_token.norm(dim=-1)
        aux_loss = 0.05 * torch.mean(torch.square(pred_norms - batch_counts))
        loss = loss + aux_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_triple.parameters(), 1.0)
    optimizer.step()

    return loss.item(), enc.get("alpha_sem"), enc.get("alpha_vis")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--traj_steps", type=int, default=30)
    parser.add_argument("--det_every", type=int, default=5)
    parser.add_argument("--skip_owl", action="store_true")
    parser.add_argument("--all", action="store_true", help="Use all 11 VLA-Arena suites")
    parser.add_argument("--suites", type=str, default=None,
                        help="Comma-separated suite short names, e.g. long_horizon,safety_state_preservation")
    parser.add_argument("--max_tasks_per_suite", type=int, default=0,
                        help="Limit tasks per suite (0=all). Useful for smoke test.")
    parser.add_argument("--debug", action="store_true",
                        help="Fast smoke test: 1 suite × 3 tasks × 15 steps")
    args = parser.parse_args()

    model_base, model_triple, processor, tokenizer = load_models()

    owl_pipe = None
    if not args.skip_owl:
        print("\nLoading OWL-ViT for visual grounding ...")
        owl_pipe = load_owl_vit_pipeline(device=device)

    all_bm = benchmark.get_benchmark_dict()
    va_suites = {}
    for bm_key in sorted(all_bm.keys()):
        if not bm_key.startswith("vla_arena_"): continue
        short = bm_key[len("vla_arena_"):]
        try:
            suite = all_bm[bm_key]()
            _ = suite.tasks
            va_suites[short] = suite
        except Exception as e:
            print(f"⚠️ Skip {bm_key}: {e}")

    if args.debug:
        va_suites = {k: va_suites[k] for k in list(sorted(va_suites.keys()))[:1]}
        args.max_tasks_per_suite = 3
        args.traj_steps = 15
        args.steps = 500
    elif args.suites:
        wanted = set(s.strip() for s in args.suites.split(","))
        va_suites = {k: v for k, v in va_suites.items() if k in wanted}

    print(f"\n{'='*60}")
    print(f"Collecting VLA-Arena trajectories ...")
    print(f"{'='*60}")

    all_traj = []
    total_successful = 0
    total_attempted = 0

    for short, suite in sorted(va_suites.items()):
        n = suite.n_tasks if args.max_tasks_per_suite == 0 else min(suite.n_tasks, args.max_tasks_per_suite)
        print(f"\n--- {short} ({n}/{suite.n_tasks} tasks) ---")
        for ti in range(n):
            task = suite.get_task(ti)
            t0 = time.time()
            try:
                traj = collect_trajectory_vla(
                    model_base, owl_pipe, processor, suite, ti,
                    num_steps=args.traj_steps, det_every=args.det_every,
                )
            except Exception as e:
                dt = time.time() - t0
                total_attempted += 1
                print(f"  ⚠️  {task.name[:50]:50s} EXC: {type(e).__name__}")
                continue
            dt = time.time() - t0
            total_attempted += 1
            if traj is not None and len(traj) > 3:
                all_traj.extend(traj)
                succ_marker = ""
                if len(traj) < args.traj_steps + 1:
                    succ_marker = " [DONE]"
                    total_successful += 1
                print(f"  ✅ {task.name[:50]:50s} {len(traj):3d} steps {dt:.0f}s{succ_marker}")
            else:
                print(f"  ❌ {task.name[:50]:50s} SKIP")

    print(f"\nTotal: {len(all_traj)} samples from {total_attempted} tasks, {total_successful} with early stop")
    if len(all_traj) < args.batch_size:
        print(f"❌ Not enough samples ({len(all_traj)} < {args.batch_size}). Abort.")
        return

    optimizer = torch.optim.AdamW(
        [p for p in model_triple.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5,
    )

    print(f"\n{'='*60}")
    print(f"Training triple residual ({args.steps} steps, lr={args.lr}, batch={args.batch_size})")
    print(f"{'='*60}")

    losses = []
    model_triple.train()

    for step in range(args.steps):
        indices = np.random.choice(len(all_traj), args.batch_size, replace=True)
        batch_items = [all_traj[i] for i in indices]

        images = torch.cat([b["images"] for b in batch_items], dim=0)
        image_mask = torch.cat([b["image_mask"] for b in batch_items], dim=0)
        proprio = torch.cat([b["proprio"] for b in batch_items], dim=0)
        abstract_tasks = [b["abstract_ids"] for b in batch_items]
        visual_semantics_list = [b.get("visual_semantic") for b in batch_items]
        teacher_action = torch.cat([b["teacher_action"] for b in batch_items], dim=0)

        raw_input_ids = [b["input_ids"].squeeze(0) for b in batch_items]
        max_text_len = max(len(x) for x in raw_input_ids)
        input_ids_padded = torch.zeros(len(batch_items), max_text_len, dtype=torch.long)
        for i, x in enumerate(raw_input_ids):
            input_ids_padded[i, :len(x)] = x

        with torch.no_grad():
            teacher_action_norm = model_triple.action_space.normalize_action(teacher_action)
            proprio_norm = model_triple.action_space.normalize_state(proprio) \
                if hasattr(model_triple.action_space, 'normalize_state') else proprio

        batch = {
            "images": images,
            "image_mask": image_mask,
            "proprio_norm": proprio_norm,
            "teacher_action_norm": teacher_action_norm,
            "abstract_tasks": abstract_tasks,
            "input_ids": input_ids_padded,
            "visual_semantics": visual_semantics_list,
        }

        loss, alpha_sem, alpha_vis = train_step(model_triple, optimizer, batch, tokenizer)
        losses.append(loss)

        if (step + 1) % 50 == 0:
            avg = np.mean(losses[-50:])
            a_s = alpha_sem.item() if alpha_sem is not None else 0
            a_v = alpha_vis.item() if alpha_vis is not None else 0
            print(f"  Step {step+1:4d}/{args.steps}  loss={avg:.6f}  α_sem={a_s:.4f}  α_vis={a_v:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    sem_state = {k: v.cpu() for k, v in model_triple.state_dict().items()
                 if "semantic" in k or "visual" in k}
    torch.save(sem_state, os.path.join(OUT_DIR, "semantic_encoder.pt"))

    final_alpha_sem = torch.sigmoid(model_triple.semantic_alpha).item()
    final_alpha_vis = torch.sigmoid(model_triple.visual_alpha).item()
    print(f"\n✅ Semantic+Visual encoder saved → {OUT_DIR}/semantic_encoder.pt")
    print(f"   Keys: {len(sem_state)}")
    print(f"   Final α_sem={final_alpha_sem:.4f}")
    print(f"   Final α_vis={final_alpha_vis:.4f}")

    config = SmolVLMVLAConfig.from_pretrained(CHECKPOINT)
    config.use_semantic = True
    config.use_visual_semantic = True
    config.semantic_hidden_size = 256
    config.semantic_num_layers = 2
    config.semantic_num_heads = 4
    config.semantic_max_len = 16
    config.semantic_vocab_size = tokenizer.vocab_size
    config.save_pretrained(OUT_DIR)
    print(f"   Config saved → {OUT_DIR}")

    print(f"\nDone. Run eval with:")
    print(f"  python eval_vla_arena.py --mode baseline  --suites long_horizon,safety_state_preservation --num_trials 5")
    print(f"  Then edit eval_results/compare.py to include both old and new results")


if __name__ == "__main__":
    main()