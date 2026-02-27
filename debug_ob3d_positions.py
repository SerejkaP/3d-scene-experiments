"""
Debug script: render one OB3D camera from a GS scene of a DIFFERENT panorama,
testing all candidate center_pos formulas.

Usage:
    python debug_ob3d_positions.py \
        --ply      /mnt/d/outputs/worldgen/ob3d/archiviz-flat/Non-Egocentric/00002_render.ply \
        --cam      /path/to/test/ob3d/archiviz-flat/Non-Egocentric/data/00010_back.json \
        --ply_cam  /path/to/test/ob3d/archiviz-flat/Non-Egocentric/cameras/00002_cam.json \
        --tgt_cam  /path/to/test/ob3d/archiviz-flat/Non-Egocentric/cameras/00010_cam.json \
        --gt       /path/to/test/ob3d/archiviz-flat/Non-Egocentric/data/00010_back.png \
        --output   /tmp/ob3d_pos_debug \
        [--swap_yz]

Variants:
    v0_origin       center = [0, 0, 0]
    v1_world10      center = world_pos_10   (world position of target pano)
    v2_world02      center = world_pos_02   (world position of source pano)
    v3_delta        center = world_pos_10 - world_pos_02   (relative, no rotation)
    v4_transform    center = R02 @ world_pos_10 + t02   (full w2c transform = theoretical)
    v5_neg_delta    center = -(world_pos_10 - world_pos_02)
    v6_neg_transform center = -(R02 @ world_pos_10 + t02)
"""

import argparse
import json
import os

import cv2
import numpy as np

from utils.gsplat_utils import render_virtual_camera


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(out, text, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
    return out


def load_pano_extrinsics(cam_json_path):
    """Load R_w2c (3x3) and t_w2c (3,) from a cameras/{id}_cam.json file."""
    with open(cam_json_path) as fh:
        d = json.load(fh)[0]
    R = np.array(d["extrinsics"]["rotation"], dtype=np.float64)
    t = np.array(d["extrinsics"]["translation"], dtype=np.float64)
    return R, t


def world_pos(R_w2c, t_w2c):
    """Camera center in world space from world-to-camera extrinsics."""
    return -R_w2c.T @ t_w2c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply",     required=True, help="GS scene PLY (e.g. 00002_render.ply)")
    parser.add_argument("--cam",     required=True, help="Target camera JSON (e.g. data/00010_back.json)")
    parser.add_argument("--ply_cam", required=True, help="cameras/{ply_id}_cam.json  (source pano extrinsics)")
    parser.add_argument("--tgt_cam", required=True, help="cameras/{tgt_id}_cam.json  (target pano extrinsics)")
    parser.add_argument("--gt",      default=None,  help="Ground-truth image for comparison (optional)")
    parser.add_argument("--output",  required=True, help="Output directory")
    parser.add_argument("--swap_yz", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Extrinsics of the source pano (the one whose GS we render from) ────────
    R02, t02 = load_pano_extrinsics(args.ply_cam)
    wp02 = world_pos(R02, t02)

    # ── Extrinsics of the target pano (the one whose camera we want) ───────────
    R10, t10 = load_pano_extrinsics(args.tgt_cam)
    wp10 = world_pos(R10, t10)

    # ── Camera intrinsics / rotation from the data JSON ────────────────────────
    with open(args.cam) as fh:
        cam_info = json.load(fh)
    cam = {
        "K":      np.array(cam_info["K"]),
        "R":      np.array(cam_info["R"]),
        "width":  cam_info["width"],
        "height": cam_info["height"],
    }
    # center_pos stored in the JSON (usually [0,0,0] for own scene)
    stored_center = np.array(cam_info.get("center_pos", [0.0, 0.0, 0.0]), dtype=np.float64)

    delta = wp10 - wp02  # relative offset in world space

    # ── All candidate center_pos values ───────────────────────────────────────
    variants: dict[str, np.ndarray] = {
        "v0_origin":        np.zeros(3),
        "v1_stored":        stored_center,                    # from JSON field
        "v2_world10":       wp10,                             # world pos of target pano
        "v3_world02":       wp02,                             # world pos of source pano
        "v4_delta":         delta,                            # relative, no rotation
        "v5_transform":     R02 @ wp10 + t02,                 # full w2c transform ← theoretical
        "v6_neg_delta":     -delta,
        "v7_neg_transform": -(R02 @ wp10 + t02),
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Source pano  world_pos: {wp02.tolist()}")
    print(f"Target pano  world_pos: {wp10.tolist()}")
    print(f"Delta (world):          {delta.tolist()}")
    print(f"v5_transform:           {(R02 @ wp10 + t02).tolist()}")
    print()

    row_imgs = []

    # Ground truth image (if provided)
    if args.gt and os.path.exists(args.gt):
        gt_img = cv2.imread(args.gt)
        if gt_img is not None:
            row_imgs.append(_label(gt_img, "GT"))

    # Render each variant
    for vname, center_pos in variants.items():
        out_path = os.path.join(args.output, f"{vname}.png")
        render_virtual_camera(
            ply_path=args.ply,
            camera=cam,
            output_path=out_path,
            center_pos=center_pos.tolist(),
            swap_yz=args.swap_yz,
        )
        pos_str = f"[{center_pos[0]:.3f}, {center_pos[1]:.3f}, {center_pos[2]:.3f}]"
        print(f"  {vname:20s}  center = {pos_str}  →  {out_path}")

        img = cv2.imread(out_path)
        if img is not None:
            row_imgs.append(_label(img, vname))

    # ── Comparison strip ───────────────────────────────────────────────────────
    if row_imgs:
        target_h = row_imgs[0].shape[0]
        resized = []
        for img in row_imgs:
            h, w = img.shape[:2]
            resized.append(cv2.resize(img, (int(w * target_h / h), target_h)))
        strip = np.concatenate(resized, axis=1)
        strip_path = os.path.join(args.output, "comparison.png")
        cv2.imwrite(strip_path, strip)
        print(f"\nComparison strip: {strip_path}")

    print(f"Done. All outputs in: {args.output}")


if __name__ == "__main__":
    main()
