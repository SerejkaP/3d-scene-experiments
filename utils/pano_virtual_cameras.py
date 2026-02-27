"""
Universal utility for converting equirectangular panoramas into sets of perspective views.

Coordinate convention (matches gsplat cubemap rendering in gsplat_utils.py):
  - Y-up world coordinate system
  - yaw=0°, pitch=0°  → camera looks towards -Z world (center of panorama)
  - yaw=90°, pitch=0° → camera looks towards -X world (90° right of center)
  - yaw=180°           → camera looks towards +Z world (rear)
  - pitch > 0°         → camera tilts upward
  - pitch < 0°         → camera tilts downward

The yaw convention is consistent with py360convert's `e2p(u_deg=yaw, v_deg=pitch)`.

Typical usage::

    cameras = get_virtual_cameras(yaw_step=60, pitch_angles=[-30, 0, 30])

    # Crop GT perspectives from a panorama
    gt_views = pano_to_perspective_views(gt_pano, cameras)

    # Render GS perspectives with matching camera params — see gsplat_utils.render_virtual_camera()
"""

import numpy as np
import cv2
import py360convert
from typing import Dict, List, Sequence


def get_virtual_cameras(
    yaw_step: float = 60.0,
    pitch_angles: Sequence[float] = (0.0,),
    fov_h: float = 90.0,
    output_width: int = 512,
    output_height: int = 512,
) -> List[Dict]:
    """
    Generate a uniform grid of virtual cameras covering the panorama sphere.

    Cameras are placed at every ``yaw_step`` degrees around the horizontal circle
    (0° … 360° exclusive) for each requested pitch level.  All cameras share the
    same field of view and resolution.

    Args:
        yaw_step:      Degrees between successive yaw samples.
                       E.g. 60 → 6 cameras per pitch level (0°, 60°, …, 300°).
        pitch_angles:  Sequence of pitch angles in degrees to sample.
                       E.g. (-30, 0, 30) → 3 rings, equatorial + two tilted.
        fov_h:         Horizontal field of view in degrees (same for all cameras).
        output_width:  Width of each perspective output image in pixels.
        output_height: Height of each perspective output image in pixels.

    Returns:
        List of camera dicts, each containing:

        * ``yaw``    – float, horizontal angle (degrees).
        * ``pitch``  – float, vertical angle (degrees).
        * ``fov_h``  – float, horizontal FOV (degrees).
        * ``fov_v``  – float, vertical FOV (degrees), derived from fov_h + aspect ratio.
        * ``width``  – int, output image width.
        * ``height`` – int, output image height.
        * ``K``      – np.ndarray (3, 3), pinhole intrinsic matrix.
        * ``R``      – np.ndarray (3, 3), world-to-camera rotation (rows = right, up, back).
    """
    fov_h_rad = np.deg2rad(fov_h)
    fov_v_rad = 2.0 * np.arctan(np.tan(fov_h_rad / 2.0) * output_height / output_width)
    fov_v = np.rad2deg(fov_v_rad)

    fx = output_width / (2.0 * np.tan(fov_h_rad / 2.0))
    fy = output_height / (2.0 * np.tan(fov_v_rad / 2.0))
    cx = output_width / 2.0
    cy = output_height / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    yaw_angles = np.arange(0.0, 360.0, yaw_step).tolist()

    cameras = []
    for pitch in pitch_angles:
        for yaw in yaw_angles:
            R = _yaw_pitch_to_rotation(yaw, pitch)
            cameras.append(
                {
                    "yaw": float(yaw),
                    "pitch": float(pitch),
                    "fov_h": float(fov_h),
                    "fov_v": float(fov_v),
                    "width": int(output_width),
                    "height": int(output_height),
                    "K": K.copy(),
                    "R": R,
                }
            )

    return cameras


def _yaw_pitch_to_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """
    Convert yaw / pitch angles to a world-to-camera rotation matrix.

    Args:
        yaw_deg:   Horizontal angle in degrees.  0 = panorama center (-Z world).
                   Positive values rotate clockwise when viewed from above.
        pitch_deg: Vertical angle in degrees.  0 = equator.  Positive = up.

    Returns:
        R: np.ndarray (3, 3).  Row layout: [right | up | back] (i.e. camera axes
           expressed in world coordinates), which is the convention used by gsplat's
           ``rasterization()`` via the view matrix::

               viewmat[:3, :3] = R
               viewmat[:3, 3]  = -R @ center_pos
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # Camera forward direction in world space.
    # yaw=0   → forward = (0, 0, -1)   matches gsplat "front" cubemap face
    # yaw=90  → forward = (-1, 0,  0)  matches gsplat "right" cubemap face
    # yaw=180 → forward = (0,  0,  1)  matches gsplat "back"  cubemap face
    forward = np.array(
        [
            -np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            -np.cos(yaw) * np.cos(pitch),
        ]
    )

    world_up = np.array([0.0, 1.0, 0.0])

    # Degenerate: camera pointing straight up or down → choose a stable fallback up.
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 0.0, 1.0]) if forward[1] > 0 else np.array([0.0, 0.0, -1.0])

    z = -forward  # "back" axis (camera looks along -z in camera space)
    x = np.cross(world_up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    return np.stack([x, y, z], axis=0)  # shape (3, 3)


def crop_perspective_from_pano(
    pano: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_h: float,
    output_width: int,
    output_height: int,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Crop a single perspective view from an equirectangular panorama.

    Args:
        pano:          Equirectangular image, shape (H, W) or (H, W, C).
                       Any dtype; the output shares the same dtype.
        yaw_deg:       Horizontal viewing angle in degrees (0 = panorama centre).
        pitch_deg:     Vertical viewing angle in degrees (0 = equator, positive = up).
        fov_h:         Horizontal field of view in degrees.
        output_width:  Output image width in pixels.
        output_height: Output image height in pixels.
        interpolation: Sampling mode passed to py360convert: ``"bilinear"`` or
                       ``"nearest"``.

    Returns:
        Perspective crop, shape (output_height, output_width [, C]), same dtype as
        ``pano``.
    """
    fov_v = float(
        np.degrees(
            2.0 * np.arctan(np.tan(np.radians(fov_h) / 2.0) * output_height / output_width)
        )
    )

    view = py360convert.e2p(
        pano,
        fov_deg=(fov_h, fov_v),
        u_deg=yaw_deg,
        v_deg=pitch_deg,
        out_hw=(output_height, output_width),
        in_rot_deg=0,
        mode=interpolation,
    )

    return view.astype(pano.dtype)


def pano_to_perspective_views(
    pano: np.ndarray,
    cameras: List[Dict],
    interpolation: str = "bilinear",
) -> List[np.ndarray]:
    """
    Convert an equirectangular panorama into a list of perspective views.

    Args:
        pano:          Equirectangular image (H, W) or (H, W, C).
        cameras:       List of camera dicts produced by :func:`get_virtual_cameras`.
        interpolation: Sampling mode: ``"bilinear"`` or ``"nearest"``.

    Returns:
        List of perspective images (output_height, output_width [, C]), one per camera,
        in the same order as ``cameras``.
    """
    return [
        crop_perspective_from_pano(
            pano=pano,
            yaw_deg=cam["yaw"],
            pitch_deg=cam["pitch"],
            fov_h=cam["fov_h"],
            output_width=cam["width"],
            output_height=cam["height"],
            interpolation=interpolation,
        )
        for cam in cameras
    ]


def save_perspective_views(
    views: List[np.ndarray],
    cameras: List[Dict],
    output_dir: str,
    prefix: str = "",
    suffix: str = "",
    ext: str = ".png",
) -> List[str]:
    """
    Save a list of perspective views to disk.

    File names follow the pattern ``{prefix}yaw{yaw:03.0f}_pitch{pitch:+03.0f}{suffix}{ext}``.

    Args:
        views:      List of images to save (RGB or BGR — whatever you pass is written as-is).
        cameras:    List of camera dicts (same order as ``views``).
        output_dir: Directory to write files into (created if absent).
        prefix:     Optional filename prefix.
        suffix:     Optional filename suffix (before extension).
        ext:        File extension including the dot.

    Returns:
        List of absolute file paths written.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for view, cam in zip(views, cameras):
        fname = f"{prefix}yaw{cam['yaw']:06.1f}_pitch{cam['pitch']:+05.1f}{suffix}{ext}"
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, view)
        paths.append(fpath)
    return paths
