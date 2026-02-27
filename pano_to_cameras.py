"""
Split an equirectangular panorama into a set of pinhole-camera perspective views.

Usage examples
--------------
# 6 cameras around the equator, 90° FOV, 512×512 output
python pano_to_cameras.py panorama.png

# Denser sampling + two extra pitch rings
python pano_to_cameras.py panorama.png --yaw-step 45 --pitch -30 0 30 --fov 90

# Wide-angle, HD output, custom save dir
python pano_to_cameras.py panorama.png --fov 120 --width 1280 --height 720 --out ./views

# Print camera intrinsics and extrinsics (no images saved)
python pano_to_cameras.py panorama.png --info-only
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(__file__))
from utils.pano_virtual_cameras import get_virtual_cameras, pano_to_perspective_views


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split an equirectangular panorama into pinhole-camera views.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("panorama", help="Path to the equirectangular panorama image.")
    p.add_argument(
        "--out", "-o",
        default=None,
        help="Output directory. Default: <panorama_stem>_views/ next to the input file.",
    )
    p.add_argument(
        "--yaw-step",
        type=float,
        default=60.0,
        metavar="DEG",
        help="Degrees between successive yaw samples (default: 60 → 6 cameras/ring).",
    )
    p.add_argument(
        "--pitch",
        type=float,
        nargs="+",
        default=[0.0],
        metavar="DEG",
        help="Pitch angle(s) in degrees (default: 0). Multiple values create extra rings.",
    )
    p.add_argument(
        "--fov",
        type=float,
        default=90.0,
        metavar="DEG",
        help="Horizontal field of view in degrees (default: 90).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=512,
        metavar="PX",
        help="Output image width in pixels (default: 512).",
    )
    p.add_argument(
        "--height",
        type=int,
        default=512,
        metavar="PX",
        help="Output image height in pixels (default: 512).",
    )
    p.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix for saved images.",
    )
    p.add_argument(
        "--ext",
        default=".png",
        choices=[".png", ".jpg", ".jpeg", ".webp"],
        help="Output image format (default: .png).",
    )
    p.add_argument(
        "--save-json",
        action="store_true",
        help="Save a cameras.json file with intrinsics and extrinsics for every view.",
    )
    p.add_argument(
        "--info-only",
        action="store_true",
        help="Print camera parameters and exit without saving images.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cam_filename(cam: dict, prefix: str, ext: str) -> str:
    return f"{prefix}yaw{cam['yaw']:06.1f}_pitch{cam['pitch']:+05.1f}{ext}"


def _print_camera_table(cameras: list) -> None:
    header = f"{'#':>3}  {'yaw':>7}  {'pitch':>7}  {'fov_h':>6}  {'fov_v':>6}  {'WxH'}"
    print(header)
    print("-" * len(header))
    for i, cam in enumerate(cameras):
        print(
            f"{i:>3}  {cam['yaw']:>7.1f}°  {cam['pitch']:>7.1f}°  "
            f"{cam['fov_h']:>6.1f}°  {cam['fov_v']:>6.1f}°  "
            f"{cam['width']}x{cam['height']}"
        )


def _cameras_to_json(cameras: list, filenames: list) -> list:
    records = []
    for cam, fname in zip(cameras, filenames):
        records.append(
            {
                "filename": fname,
                "yaw_deg": cam["yaw"],
                "pitch_deg": cam["pitch"],
                "fov_h_deg": cam["fov_h"],
                "fov_v_deg": cam["fov_v"],
                "width": cam["width"],
                "height": cam["height"],
                # K as nested list for JSON serialisability
                "K": cam["K"].tolist(),
                # R (world-to-camera rotation, rows = right, up, back)
                "R": cam["R"].tolist(),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- load panorama ---
    pano_bgr = cv2.imread(args.panorama, cv2.IMREAD_COLOR)
    if pano_bgr is None:
        sys.exit(f"Error: cannot read image '{args.panorama}'")
    pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

    H_pano, W_pano = pano_rgb.shape[:2]
    print(f"Panorama : {args.panorama}  ({W_pano}×{H_pano})")

    # --- build virtual cameras ---
    cameras = get_virtual_cameras(
        yaw_step=args.yaw_step,
        pitch_angles=args.pitch,
        fov_h=args.fov,
        output_width=args.width,
        output_height=args.height,
    )

    n_yaw = int(round(360.0 / args.yaw_step))
    print(
        f"Cameras  : {len(cameras)}  "
        f"({n_yaw} yaw × {len(args.pitch)} pitch ring(s), "
        f"FOV {args.fov}°, output {args.width}×{args.height})"
    )
    print()
    _print_camera_table(cameras)
    print()

    if args.info_only:
        return

    # --- output directory ---
    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.panorama))[0]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.panorama)), f"{stem}_views")
    else:
        out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # --- crop perspectives ---
    views = pano_to_perspective_views(pano_rgb, cameras)
    filenames = [_cam_filename(cam, args.prefix, args.ext) for cam in cameras]

    for view_rgb, cam, fname in zip(views, cameras, filenames):
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, cv2.cvtColor(view_rgb, cv2.COLOR_RGB2BGR))
        print(f"  saved  {fpath}")

    # --- optional JSON ---
    if args.save_json:
        records = _cameras_to_json(cameras, filenames)
        json_path = os.path.join(out_dir, f"{args.prefix}cameras.json")
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\n  cameras.json → {json_path}")

    print(f"\nDone. {len(views)} views saved to: {out_dir}")


if __name__ == "__main__":
    main()
