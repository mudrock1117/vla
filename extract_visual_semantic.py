#!/usr/bin/env python3
"""
Pre-extract VisualSemantic for all LIBERO training trajectories.

Pipeline:
  1. Load LIBERO trajectory HDF5 files
  2. For each timestep: run OWL-ViT on agentview_image
  3. Save VisualSemantic (detections + counts + bboxes) to disk

Run once, then train_semantic_ablation.py loads the cached VisualSemantic.
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import h5py

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "libero" / "LIBERO"))

from models.visual_grounding import (
    load_owl_vit_pipeline, detect_objects, build_visual_semantic, VisualSemantic
)


def extract_for_trajectory(pipe, hdf5_path: str, task_name: str, 
                           confidence_threshold: float = 0.05,
                           nms_threshold: float = 0.4,
                           max_steps: int = 200) -> List[Dict]:
    """Extract VisualSemantic for each frame in a trajectory HDF5."""
    results = []
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'agentview_image' not in f:
                return results
            
            images = f['agentview_image']
            num_frames = min(len(images), max_steps)
            
            for t in range(num_frames):
                img = images[t]
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                img = np.ascontiguousarray(img)
                
                dets = detect_objects(pipe, img, 
                                     confidence_threshold=confidence_threshold,
                                     nms_threshold=nms_threshold)
                vs = build_visual_semantic(dets, img.shape[0], img.shape[1])
                
                results.append({
                    "task_name": task_name,
                    "step": t,
                    "counts": {
                        "graspable_container": vs.num_graspable_containers,
                        "bowl_like": vs.num_bowl_like,
                        "food_item": vs.num_food_items,
                        "box_container": vs.num_box_containers,
                        "flat_surface": vs.num_flat_surfaces,
                        "articulated": vs.num_articulated,
                    },
                    "objects": [
                        {
                            "category": o.category,
                            "bbox": o.bbox,
                            "confidence": float(o.confidence),
                            "label_text": o.label_text,
                        }
                        for o in vs.objects[:8]
                    ],
                })
    except Exception as e:
        print(f"  ⚠️ Error reading {hdf5_path}: {e}")
    
    return results


def find_trajectory_files(data_root: str) -> List[tuple]:
    """Find all HDF5 trajectory files in the dataset."""
    trajectories = []
    
    for suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        suite_dir = Path(data_root) / suite_name
        if not suite_dir.exists():
            continue
        
        for hdf5_file in sorted(suite_dir.glob("*.hdf5")):
            task_name = hdf5_file.stem.split("_demo")[0]
            trajectories.append((str(hdf5_file), task_name))
    
    return trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, 
                       default="/media/zdh/f37daf80-9fee-4790-aa5a-bcedd13aa82e/2026/hdx/SimVLA/data/LIBERO")
    parser.add_argument("--output", type=str, default="data/visual_semantic_cache.json")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--nms", type=float, default=0.4)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="0=all, N=first N files")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading OWL-ViT pipeline on {device}...")
    pipe = load_owl_vit_pipeline(device=device)
    
    print(f"\nScanning {args.data_root} ...")
    traj_files = find_trajectory_files(args.data_root)
    print(f"Found {len(traj_files)} trajectory files")
    
    if args.limit > 0:
        traj_files = traj_files[:args.limit]
        print(f"  (limited to {args.limit})")
    
    all_results = []
    start_time = time.time()
    
    for i, (hdf5_path, task_name) in enumerate(traj_files):
        print(f"[{i+1}/{len(traj_files)}] {task_name[:50]:50s}", end=" ", flush=True)
        
        results = extract_for_trajectory(
            pipe, hdf5_path, task_name,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms,
            max_steps=args.max_steps,
        )
        all_results.extend(results)
        
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        print(f"→ {len(results)} frames ({rate:.1f} files/min)")
    
    print(f"\nTotal: {len(all_results)} VisualSemantic entries")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()