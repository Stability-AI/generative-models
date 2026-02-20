#!/usr/bin/env python3
"""
Generate deterministic synthetic geometric scenes for robotics evaluation.

Outputs:
  - RGB images with rendered checkerboard + cube wireframe.
  - JSON ground-truth labels containing:
      * camera intrinsics/extrinsics
      * 3D object points
      * projected 2D keypoints

This script is intentionally lightweight (numpy + opencv only) so it can be
used in CI or simple dataset bootstrapping workflows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def build_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    fov_rad = math.radians(fov_deg)
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return k


def look_at_rotation(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Camera coordinates: x-right, y-down, z-forward
    # Keep y-down by negating up basis.
    r_cw = np.stack([right, -up, forward], axis=0)
    return r_cw


def world_to_camera(
    points_world: np.ndarray, r_cw: np.ndarray, camera_pos: np.ndarray
) -> np.ndarray:
    # X_cam = R * (X_world - C)
    return (r_cw @ (points_world - camera_pos).T).T


def project_points(points_cam: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = z > 1e-6
    points_2d = np.full((len(points_cam), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        p = points_cam[valid]
        uv = (k @ p.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        points_2d[valid] = uv
    return points_2d, valid


def build_checkerboard_points(rows: int, cols: int, square_size: float) -> np.ndarray:
    # Checkerboard lies on z=0 plane.
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append([c * square_size, r * square_size, 0.0])
    return np.array(pts, dtype=np.float64)


def build_cube_points(center: np.ndarray, size: float, yaw_deg: float) -> np.ndarray:
    h = size / 2.0
    base = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float64,
    )
    yaw = math.radians(yaw_deg)
    cz, sz = math.cos(yaw), math.sin(yaw)
    r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (r_z @ base.T).T + center


def draw_checkerboard(
    img: np.ndarray,
    corners_uv: np.ndarray,
    rows: int,
    cols: int,
    color_a: Tuple[int, int, int],
    color_b: Tuple[int, int, int],
) -> None:
    # Fill quads for each checker cell where all 4 corners are valid.
    for r in range(rows - 1):
        for c in range(cols - 1):
            i0 = r * cols + c
            i1 = i0 + 1
            i2 = i0 + cols + 1
            i3 = i0 + cols
            quad = np.array([corners_uv[i0], corners_uv[i1], corners_uv[i2], corners_uv[i3]])
            if np.isnan(quad).any():
                continue
            poly = np.round(quad).astype(np.int32)
            color = color_a if (r + c) % 2 == 0 else color_b
            cv2.fillConvexPoly(img, poly, color, lineType=cv2.LINE_AA)


def draw_wire_cube(img: np.ndarray, cube_uv: np.ndarray, cube_valid: np.ndarray) -> None:
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        if not (cube_valid[a] and cube_valid[b]):
            continue
        pa = tuple(np.round(cube_uv[a]).astype(int))
        pb = tuple(np.round(cube_uv[b]).astype(int))
        cv2.line(img, pa, pb, (30, 220, 250), 2, cv2.LINE_AA)

    for i in range(8):
        if cube_valid[i]:
            p = tuple(np.round(cube_uv[i]).astype(int))
            cv2.circle(img, p, 3, (255, 255, 255), -1, cv2.LINE_AA)


def as_list(a: np.ndarray) -> List[Any]:
    return a.tolist()


def generate_frame(
    frame_idx: int,
    width: int,
    height: int,
    rng: np.random.Generator,
    rows: int,
    cols: int,
    square_size: float,
    cube_size: float,
    fov_deg: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    img = np.full((height, width, 3), (15, 20, 28), dtype=np.uint8)

    # Scene geometry in world frame
    checker_pts_world = build_checkerboard_points(rows, cols, square_size)
    board_center = np.array(
        [((cols - 1) * square_size) * 0.5, ((rows - 1) * square_size) * 0.5, 0.0],
        dtype=np.float64,
    )

    # Slight random object pose for variation, deterministic via seed.
    cube_center = board_center + np.array(
        [
            rng.uniform(-0.08, 0.08),
            rng.uniform(-0.08, 0.08),
            cube_size * 0.55 + rng.uniform(0.00, 0.02),
        ],
        dtype=np.float64,
    )
    cube_yaw = rng.uniform(-45.0, 45.0)
    cube_pts_world = build_cube_points(cube_center, cube_size, cube_yaw)

    # Camera orbit around board center.
    radius = rng.uniform(0.9, 1.4)
    azimuth = rng.uniform(0.0, 2.0 * math.pi)
    elevation = rng.uniform(0.35, 0.85)
    camera_pos = board_center + np.array(
        [radius * math.cos(azimuth), radius * math.sin(azimuth), elevation], dtype=np.float64
    )
    target = board_center + np.array([0.0, 0.0, 0.12], dtype=np.float64)

    r_cw = look_at_rotation(camera_pos, target)
    t_cw = -r_cw @ camera_pos
    k = build_intrinsics(width, height, fov_deg)

    checker_cam = world_to_camera(checker_pts_world, r_cw, camera_pos)
    cube_cam = world_to_camera(cube_pts_world, r_cw, camera_pos)
    checker_uv, checker_valid = project_points(checker_cam, k)
    cube_uv, cube_valid = project_points(cube_cam, k)

    draw_checkerboard(
        img,
        checker_uv,
        rows,
        cols,
        color_a=(180, 180, 180),
        color_b=(70, 70, 70),
    )
    draw_wire_cube(img, cube_uv, cube_valid)

    cv2.putText(
        img,
        f"frame {frame_idx:04d}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    label: Dict[str, Any] = {
        "frame_index": frame_idx,
        "camera": {
            "width": width,
            "height": height,
            "fov_deg": fov_deg,
            "K": as_list(k),
            "R_cw": as_list(r_cw),
            "t_cw": as_list(t_cw),
            "camera_position_world": as_list(camera_pos),
            "target_world": as_list(target),
        },
        "objects": {
            "checkerboard": {
                "rows": rows,
                "cols": cols,
                "square_size_m": square_size,
                "points_world": as_list(checker_pts_world),
                "points_cam": as_list(checker_cam),
                "points_uv": as_list(checker_uv),
                "valid_mask": checker_valid.astype(bool).tolist(),
            },
            "cube": {
                "size_m": cube_size,
                "yaw_deg": cube_yaw,
                "center_world": as_list(cube_center),
                "corners_world": as_list(cube_pts_world),
                "corners_cam": as_list(cube_cam),
                "corners_uv": as_list(cube_uv),
                "valid_mask": cube_valid.astype(bool).tolist(),
            },
        },
    }
    return img, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic geometric scenes + GT labels for robotics."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robotics_geometry"))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checker-rows", type=int, default=7)
    parser.add_argument("--checker-cols", type=int, default=10)
    parser.add_argument("--square-size", type=float, default=0.04)
    parser.add_argument("--cube-size", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_img = args.output_dir / "images"
    out_lbl = args.output_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    for i in range(args.frames):
        img, label = generate_frame(
            frame_idx=i,
            width=args.width,
            height=args.height,
            rng=rng,
            rows=args.checker_rows,
            cols=args.checker_cols,
            square_size=args.square_size,
            cube_size=args.cube_size,
            fov_deg=args.fov_deg,
        )
        img_path = out_img / f"frame_{i:04d}.png"
        lbl_path = out_lbl / f"frame_{i:04d}.json"
        cv2.imwrite(str(img_path), img)
        lbl_path.write_text(json.dumps(label, indent=2))

    summary = {
        "frames": args.frames,
        "image_size": [args.width, args.height],
        "fov_deg": args.fov_deg,
        "seed": args.seed,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

