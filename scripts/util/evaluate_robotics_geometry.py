#!/usr/bin/env python3
"""
Evaluate geometric consistency of generated robotics scene labels.

This script recomputes image projections from world points and camera parameters
stored in each frame label, then compares with stored 2D points.

Metrics:
  - mean / median / max reprojection error (pixels)
  - per-frame checkerboard and cube errors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def project(points_world: np.ndarray, r_cw: np.ndarray, t_cw: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # X_cam = R * X_world + t
    points_cam = (r_cw @ points_world.T).T + t_cw.reshape(1, 3)
    z = points_cam[:, 2]
    valid = z > 1e-8
    uv = np.full((len(points_world), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        p = points_cam[valid]
        pix = (k @ p.T).T
        pix = pix[:, :2] / pix[:, 2:3]
        uv[valid] = pix
    return uv, valid


def reprojection_error(pred_uv: np.ndarray, gt_uv: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid = valid_mask.astype(bool)
    valid &= ~np.isnan(pred_uv).any(axis=1)
    valid &= ~np.isnan(gt_uv).any(axis=1)
    if not np.any(valid):
        return np.array([], dtype=np.float64)
    d = pred_uv[valid] - gt_uv[valid]
    return np.linalg.norm(d, axis=1)


def evaluate_label(label_path: Path) -> Dict:
    data = json.loads(label_path.read_text())
    cam = data["camera"]
    obj = data["objects"]

    k = np.array(cam["K"], dtype=np.float64)
    r_cw = np.array(cam["R_cw"], dtype=np.float64)
    t_cw = np.array(cam["t_cw"], dtype=np.float64)

    cb_world = np.array(obj["checkerboard"]["points_world"], dtype=np.float64)
    cb_uv_gt = np.array(obj["checkerboard"]["points_uv"], dtype=np.float64)
    cb_valid_gt = np.array(obj["checkerboard"]["valid_mask"], dtype=bool)
    cb_uv_pred, cb_valid_pred = project(cb_world, r_cw, t_cw, k)
    cb_err = reprojection_error(cb_uv_pred, cb_uv_gt, cb_valid_gt & cb_valid_pred)

    cube_world = np.array(obj["cube"]["corners_world"], dtype=np.float64)
    cube_uv_gt = np.array(obj["cube"]["corners_uv"], dtype=np.float64)
    cube_valid_gt = np.array(obj["cube"]["valid_mask"], dtype=bool)
    cube_uv_pred, cube_valid_pred = project(cube_world, r_cw, t_cw, k)
    cube_err = reprojection_error(cube_uv_pred, cube_uv_gt, cube_valid_gt & cube_valid_pred)

    all_err = np.concatenate([cb_err, cube_err]) if (len(cb_err) or len(cube_err)) else np.array([], dtype=np.float64)

    def stats(v: np.ndarray) -> Dict[str, float]:
        if len(v) == 0:
            return {"count": 0, "mean": float("nan"), "median": float("nan"), "max": float("nan")}
        return {
            "count": int(len(v)),
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "max": float(np.max(v)),
        }

    return {
        "frame_index": int(data["frame_index"]),
        "checkerboard": stats(cb_err),
        "cube": stats(cube_err),
        "all": stats(all_err),
    }


def aggregate(frame_results: List[Dict]) -> Dict:
    all_values = []
    cb_values = []
    cube_values = []
    for r in frame_results:
        if r["all"]["count"] > 0 and not np.isnan(r["all"]["mean"]):
            all_values.append(r["all"]["mean"])
        if r["checkerboard"]["count"] > 0 and not np.isnan(r["checkerboard"]["mean"]):
            cb_values.append(r["checkerboard"]["mean"])
        if r["cube"]["count"] > 0 and not np.isnan(r["cube"]["mean"]):
            cube_values.append(r["cube"]["mean"])

    def agg(v: List[float]) -> Dict[str, float]:
        if not v:
            return {"frames": 0, "mean_of_means": float("nan"), "median_of_means": float("nan"), "max_of_means": float("nan")}
        a = np.array(v, dtype=np.float64)
        return {
            "frames": int(len(a)),
            "mean_of_means": float(np.mean(a)),
            "median_of_means": float(np.median(a)),
            "max_of_means": float(np.max(a)),
        }

    return {"all": agg(all_values), "checkerboard": agg(cb_values), "cube": agg(cube_values)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate reprojection consistency for generated robotics scene labels.")
    p.add_argument("--dataset-dir", type=Path, required=True, help="Path like outputs/robotics_geometry")
    p.add_argument("--max-mean-error-px", type=float, default=0.25, help="Fail if global mean_of_means exceeds this threshold")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels_dir = args.dataset_dir / "labels"
    if not labels_dir.exists():
        raise SystemExit(f"labels directory not found: {labels_dir}")

    label_files = sorted(labels_dir.glob("frame_*.json"))
    if not label_files:
        raise SystemExit(f"no frame_*.json labels found in: {labels_dir}")

    frame_results = [evaluate_label(p) for p in label_files]
    summary = aggregate(frame_results)

    out = {
        "dataset_dir": str(args.dataset_dir),
        "frames_evaluated": len(frame_results),
        "threshold_max_mean_error_px": args.max_mean_error_px,
        "summary": summary,
    }
    print(json.dumps(out, indent=2))

    global_mean = summary["all"]["mean_of_means"]
    if not np.isnan(global_mean) and global_mean > args.max_mean_error_px:
        raise SystemExit(
            f"FAILED: global mean reprojection error {global_mean:.6f}px > {args.max_mean_error_px:.6f}px"
        )


if __name__ == "__main__":
    main()

