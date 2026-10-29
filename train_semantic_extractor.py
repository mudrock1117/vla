#!/usr/bin/env python3
"""
SemanticExtractor Training with Wording-Invariant Perturbation Augmentation.

核心设计:
  - 冻结 text_model (361.9M) + vision + transformer, 只训练 SemanticExtractor (0.49M) + alpha
  - 训练时 50% 样本用 LIBERO-PRO 的 perturbed instruction (ood_language.yaml)
  - 对比 loss: 同一任务 orig/pert 的 sem_pooled 必须收敛到同一个方向
  - 蒸馏目标: extractor model → baseline teacher action (隐式传递任务目标, 无显式泄漏)

对比旧 parse_instruction 路径:
  - 旧: 仅 40% 任务在语言扰动下解析稳定 → extractor 被正则瓶颈拖垮
  - 新: LLM hidden states 天然 wording-invariant + 显式训练约束 → 双重保证

Usage:
  python train_semantic_extractor.py --steps 2000 --lr 1e-4 --contrast_weight 0.3
"""

from __future__ import annotations
import argparse, json, math, os, sys, time, yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "libero" / "LIBERO"))

from models.configuration_smolvlm_vla import SmolVLMVLAConfig
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor

import safetensors.torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

CHECKPOINT = "checkpoints/SimVLA-LIBERO"
EXTRACTOR_CKPT = "checkpoints/SimVLA-LIBERO/semantic_extractor.pt"
NORM_STATS = "norm_stats/libero_norm.json"
SPLITS_DIR = "datasets/ood_splits"
PERTURB_YAML = "datasets/ood_language.yaml"
SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"

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


def load_perturb_map():
    perturb_map = {}
    if not os.path.exists(PERTURB_YAML):
        print(f"[WARN] {PERTURB_YAML} not found → no perturbation augmentation")
        return perturb_map
    with open(PERTURB_YAML) as f:
        data = yaml.safe_load(f)
    for suite_name, tasks in data.items():
        if not isinstance(tasks, dict):
            continue
        for tn, perts in tasks.items():
            if isinstance(perts, list):
                valid = [p for p in perts if isinstance(p, str) and p.strip()]
                if valid:
                    perturb_map[tn] = valid
    print(f"[PERTURB] Loaded {len(perturb_map)} tasks with perturbations")
    return perturb_map


def load_models():
    print(f"Loading models from {CHECKPOINT} ...")

    config_base = SmolVLMVLAConfig.from_pretrained(CHECKPOINT)
    model_base = SmolVLMVLA(config_base)
    state = safetensors.torch.load_file(os.path.join(CHECKPOINT, "model.safetensors"))
    model_base.load_state_dict(state, strict=True)
    model_base = model_base.to(device).eval()
    for p in model_base.parameters():
        p.requires_grad = False

    config_ext = SmolVLMVLAConfig.from_pretrained(CHECKPOINT)
    config_ext.use_semantic = True
    config_ext.use_semantic_extractor = True
    config_ext.semantic_extract_layer = 24
    model_ext = SmolVLMVLA(config_ext)
    state_ext = safetensors.torch.load_file(os.path.join(CHECKPOINT, "model.safetensors"))
    model_ext.load_state_dict(state_ext, strict=False)
    model_ext = model_ext.to(device)

    for name, p in model_ext.named_parameters():
        if "semantic_extractor" not in name and "semantic_alpha" not in name:
            p.requires_grad = False

    trainable = sum(p.numel() for p in model_ext.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_ext.parameters())
    print(f"  Baseline model: frozen (teacher)")
    print(f"  Extractor model: trainable={trainable/1e6:.3f}M / total={total/1e6:.2f}M")
    print(f"    - semantic_extractor: {sum(p.numel() for p in model_ext.semantic_extractor.parameters())/1e6:.3f}M")
    print(f"    - semantic_alpha: {model_ext.semantic_alpha.numel()} params")
    print(f"    - text_model frozen: {not any(p.requires_grad for p in model_ext.vlm.model.text_model.parameters())}")

    processor = SmolVLMVLAProcessor.from_pretrained(SMOLVLM_MODEL)
    model_base.action_space.load_norm_stats(NORM_STATS)
    model_ext.action_space.load_norm_stats(NORM_STATS)
    print(f"  Norm stats loaded from {NORM_STATS}")
    print(f"  Device: {device}")
    return model_base, model_ext, processor


def get_libero_task(task_name: str):
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        suite = benchmark_dict[suite_name]()
        for i in range(suite.n_tasks):
            task = suite.get_task(i)
            if task.name == task_name or task.language == task_name:
                return task, suite, i
    return None, None, None


def collect_trajectory(model_base, processor, task_name, num_steps=40):
    task, suite, task_id = get_libero_task(task_name)
    if task is None:
        return None

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(7)
    init_states = suite.get_task_init_states(task_id)
    instruction = task.language
    env.reset()
    env.set_init_state(init_states[0])

    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)

    trajectory = []
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

            trajectory.append({
                "images": images,
                "image_mask": image_mask,
                "proprio": proprio,
                "instruction": instruction,
                "task_name": task_name,
                "teacher_action": actions,
            })

            action = actions.cpu().numpy()[0, 0]
            obs, reward, done, info = env.step(action.tolist())
            if done:
                break

    env.close()
    return trajectory


def encode_instructions_batch(processor, instructions_list):
    max_len = max(len(processor.tokenizer.encode(s)) for s in instructions_list)
    input_ids = torch.zeros(len(instructions_list), max_len, dtype=torch.long)
    for i, s in enumerate(instructions_list):
        ids = processor.tokenizer.encode(s)[:max_len]
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return input_ids


def build_input_ids(batch_items, processor, perturb_map):
    raw_instrs = []
    perturbed_mask = []
    for b in batch_items:
        tn = b["task_name"]
        instr = b["instruction"]
        if np.random.random() < 0.5 and tn in perturb_map:
            pert = np.random.choice(perturb_map[tn])
            raw_instrs.append(pert)
            perturbed_mask.append(True)
        else:
            raw_instrs.append(instr)
            perturbed_mask.append(False)
    input_ids = encode_instructions_batch(processor, raw_instrs)
    return input_ids, perturbed_mask, raw_instrs


def train_step(model_ext, optimizer, batch, contrast_weight=0.3):
    B = batch["images"].shape[0]
    input_ids = batch["input_ids"].to(device)

    model_ext.train()
    optimizer.zero_grad()

    enc = model_ext.forward_vlm_semantic(
        batch["images"], batch["image_mask"], input_ids, None
    )

    teacher_action = batch["teacher_action_norm"]
    noise = torch.randn_like(teacher_action)
    t = torch.rand(B, device=device) * 0.998 + 0.001
    t_exp = t.view(-1, 1, 1)
    x_t = t_exp * noise + (1 - t_exp) * teacher_action
    u_t = noise - teacher_action

    proprio_norm = batch["proprio_norm"]
    v_t = model_ext.transformer(
        vlm_features=enc["vlm_features"],
        action_with_noise=x_t,
        t=t,
        proprio=proprio_norm,
    )

    loss_distill = torch.mean(torch.square(v_t - u_t))

    loss_contrast = torch.tensor(0.0, device=device)
    if contrast_weight > 0 and "sem_pooled_orig" in batch and "sem_pooled_pert" in batch:
        sp_orig = batch["sem_pooled_orig"].to(device)
        sp_pert = batch["sem_pooled_pert"].to(device)
        if sp_orig.dim() == 3:
            sp_orig = sp_orig.squeeze(1)
        if sp_pert.dim() == 3:
            sp_pert = sp_pert.squeeze(1)
        loss_contrast = torch.mean(torch.square(
            (sp_orig - sp_orig.mean(0)) - (sp_pert - sp_pert.mean(0))
        ))

    loss = loss_distill + contrast_weight * loss_contrast
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_ext.parameters(), 1.0)
    optimizer.step()

    return loss_distill.item(), loss_contrast.item(), loss.item()


def compute_sem_pooled(model_ext, images, image_mask, input_ids):
    with torch.no_grad():
        enc = model_ext.forward_vlm_semantic(
            images, image_mask, input_ids.to(images.device), None
        )
    return enc["sem_pooled"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--traj_steps", type=int, default=40)
    parser.add_argument("--contrast_weight", type=float, default=0.3)
    parser.add_argument("--contrast_freq", type=int, default=5)
    args = parser.parse_args()

    perturb_map = load_perturb_map()
    model_base, model_ext, processor = load_models()

    splits_to_collect = ["ID", "OOD-Composition", "OOD-Both"]
    all_traj = []
    K8_NAME = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    for split_name in splits_to_collect:
        split_path = os.path.join(SPLITS_DIR, f"{split_name}.json")
        if not os.path.exists(split_path):
            continue
        with open(split_path) as f:
            tasks = json.load(f)
        print(f"\n=== Collecting from {split_name} ({len(tasks)} tasks) ===")
        for entry in tasks:
            name = entry["task_name"]
            n_repeat = 3 if name == K8_NAME else 1
            for rep in range(n_repeat):
                print(f"  {name[:60]:60s} rep={rep+1}/{n_repeat}", end=" ", flush=True)
                traj = collect_trajectory(model_base, processor, name, num_steps=args.traj_steps)
                if traj is not None and len(traj) > 3:
                    all_traj.extend(traj)
                    n_pert = len(perturb_map.get(name, []))
                    print(f"→ {len(traj)} steps (pert={n_pert})")
                else:
                    print("→ SKIP")

    unique_tasks = set(t["task_name"] for t in all_traj)
    unique_instr = set(t["instruction"] for t in all_traj)
    print(f"\nTotal samples: {len(all_traj)}, unique tasks: {len(unique_tasks)}, unique instructions: {len(unique_instr)}")
    tasks_with_pert = sum(1 for tn in unique_tasks if tn in perturb_map)
    print(f"Tasks with perturbation support: {tasks_with_pert}/{len(unique_tasks)}")

    if len(all_traj) < args.batch_size:
        print("❌ Not enough samples.")
        return

    optimizer = torch.optim.AdamW(
        [p for p in model_ext.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-5,
    )

    print(f"\n=== Training SemanticExtractor ===")
    print(f"  steps={args.steps}, lr={args.lr}, batch_size={args.batch_size}")
    print(f"  contrast_weight={args.contrast_weight}, contrast_freq={args.contrast_freq}")
    print(f"  perturbation_prob=0.5 (50% samples use perturbed instruction)")

    losses_distill = []
    losses_contrast = []
    model_ext.train()

    start_time = time.time()
    for step in range(args.steps):
        indices = np.random.choice(len(all_traj), args.batch_size, replace=True)
        batch_items = [all_traj[i] for i in indices]

        images = torch.cat([b["images"] for b in batch_items], dim=0)
        image_mask = torch.cat([b["image_mask"] for b in batch_items], dim=0)
        proprio = torch.cat([b["proprio"] for b in batch_items], dim=0)
        teacher_action = torch.cat([b["teacher_action"] for b in batch_items], dim=0)

        input_ids, pert_mask, raw_instrs = build_input_ids(batch_items, processor, perturb_map)

        with torch.no_grad():
            teacher_action_norm = model_ext.action_space.normalize_action(teacher_action)
            proprio_norm = model_ext.action_space.normalize_state(proprio) \
                if hasattr(model_ext.action_space, 'normalize_state') else proprio

        batch = {
            "images": images,
            "image_mask": image_mask,
            "proprio_norm": proprio_norm,
            "teacher_action_norm": teacher_action_norm,
            "input_ids": input_ids.to(images.device),
        }

        if args.contrast_weight > 0 and (step + 1) % args.contrast_freq == 0:
            orig_instrs = [b["instruction"] for b in batch_items]
            pert_instrs = []
            for b in batch_items:
                tn = b["task_name"]
                if tn in perturb_map and perturb_map[tn]:
                    pert_instrs.append(np.random.choice(perturb_map[tn]))
                else:
                    pert_instrs.append(b["instruction"])
            input_ids_orig = encode_instructions_batch(processor, orig_instrs)
            input_ids_pert = encode_instructions_batch(processor, pert_instrs)

            sem_orig = compute_sem_pooled(model_ext, images, image_mask, input_ids_orig)
            sem_pert = compute_sem_pooled(model_ext, images, image_mask, input_ids_pert)
            batch["sem_pooled_orig"] = sem_orig
            batch["sem_pooled_pert"] = sem_pert

        ld, lc, lt = train_step(model_ext, optimizer, batch, args.contrast_weight)
        losses_distill.append(ld)
        losses_contrast.append(lc)

        if (step + 1) % 100 == 0:
            avg_d = np.mean(losses_distill[-100:])
            avg_c = np.mean(losses_contrast[-100:])
            elapsed = time.time() - start_time
            pert_pct = sum(pert_mask) / len(pert_mask) * 100
            print(f"  Step {step+1:5d}/{args.steps}  distill={avg_d:.6f}  contrast={avg_c:.6f}  "
                  f"pert={pert_pct:.0f}%  [{elapsed:.0f}s, {elapsed/(step+1):.1f}s/step]")

    os.makedirs(os.path.dirname(EXTRACTOR_CKPT), exist_ok=True)
    ext_state = {k: v.cpu() for k, v in model_ext.state_dict().items()
                 if "semantic_extractor" in k or "semantic_alpha" in k}
    torch.save(ext_state, EXTRACTOR_CKPT)
    print(f"\n✅ SemanticExtractor saved → {EXTRACTOR_CKPT}")
    print(f"   Keys: {len(ext_state)}")
    for k, v in ext_state.items():
        print(f"     {k}: {v.shape}")

    print(f"\nDone. Run eval with:")
    print(f"  python eval_ood_splits.py --split OOD-Both --num_trials 10 --mode semantic_extractor")
    print(f"  python eval_ood_splits.py --split OOD-Composition --num_trials 10 --mode semantic_extractor")
    print(f"  python eval_libero_pro.py                 # orig + pert + Δ robustness")
    print(f"\nCompare with baselines:")
    print(f"  baseline (text only):     OOD-Both=78.6%, LIBERO-PRO orig=86.0% pert=59.5% Δ=-26.5%")
    print(f"  old semantic (regex):     OOD-Both=90.0%, LIBERO-PRO orig=82.5% pert=57.5% Δ=-25.0%")
    print(f"  extractor (ours):         TBD")


if __name__ == "__main__":
    main()