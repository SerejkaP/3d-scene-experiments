import math
import torch
import numpy as np
from plyfile import PlyData
import cv2
import os
from typing import Dict, Optional, Sequence, Tuple
from gsplat.rendering import rasterization
import py360convert
import OpenEXR
import Imath


def compute_scene_radius(ply_path: str, percentile: float = 90.0) -> float:
    """
    Estimate the spatial extent of a Gaussian Splatting scene.

    Computes the given percentile of the Euclidean distance of each Gaussian
    mean from the scene origin.  Useful for auto-scaling camera positions when
    the GS coordinate space and the dataset world-coordinate space are at
    different scales (e.g. DreamScene360 vs. WorldGen).

    Args:
        ply_path:   Path to the Gaussian Splatting PLY file.
        percentile: Percentile of mean distances to return (default 90).

    Returns:
        Scene radius estimate in GS coordinate units.
    """
    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    dists = np.linalg.norm(means, axis=-1)
    return float(np.percentile(dists, percentile))


def _save_depth_exr(depth_map: np.ndarray, output_path: str, channel: str = "Z"):
    height, width = depth_map.shape
    header = OpenEXR.Header(width, height)
    depth_float32 = depth_map.astype(np.float32)
    depth_bytes = depth_float32.tobytes()
    header["channels"] = {
        channel: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    exr_file = OpenEXR.OutputFile(output_path, header)
    exr_file.writePixels({channel: depth_bytes})
    exr_file.close()


def load_depth_exr(
    exr_path: str,
    channel: str = None,
    fallback_channels: list = ["Z", "depth", "R", "V", "distance", "G", "B"],
) -> np.ndarray:
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    available_channels = header["channels"].keys()
    if channel is not None:
        if channel not in available_channels:
            raise ValueError(
                f"Channel '{channel}' not found in EXR. Available: {list(available_channels)}"
            )
        depth_channel = channel
    else:
        depth_channel = None
        for ch in fallback_channels:
            if ch in available_channels:
                depth_channel = ch
                break
        if depth_channel is None:
            raise ValueError(
                f"No depth channel found in EXR. Available: {list(available_channels)}"
            )
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_bytes = exr_file.channel(depth_channel, pt)
    depth_map = np.frombuffer(depth_bytes, dtype=np.float32)
    depth_map = depth_map.reshape(height, width)

    return depth_map


def load_depth(depth_path: str, depth_scale: float = 1.0) -> np.ndarray:
    """
    Load depth map from various formats (EXR, PNG, NPY).

    Args:
        depth_path: Path to depth file
        depth_scale: Scale factor to convert to meters
            - For PNG in millimeters: 1/1000.0
            - For PNG in meters: 1.0
            - For EXR/NPY already in meters: 1.0

    Returns:
        Depth map as numpy array (H, W) in meters
    """
    if depth_path.lower().endswith(".exr"):
        return load_depth_exr(depth_path)
    elif depth_path.lower().endswith(".npy"):
        return np.load(depth_path).astype(np.float32)
    else:
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Could not load depth from {depth_path}")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return depth.astype(np.float32) * depth_scale


def _load_gs_ply(path, device):
    plydata = PlyData.read(path)
    v = plydata["vertex"]

    means = torch.tensor(
        np.stack([v["x"], v["y"], v["z"]], axis=-1), device=device, dtype=torch.float32
    )
    colors_dc = torch.tensor(
        np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1),
        device=device,
        dtype=torch.float32,
    )
    colors = 0.5 + 0.28209479177387814 * colors_dc

    opacity = torch.tensor(v["opacity"], device=device, dtype=torch.float32)
    opacity = torch.sigmoid(opacity)

    scales = torch.tensor(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1),
        device=device,
        dtype=torch.float32,
    )
    scales = torch.exp(scales)

    quats = torch.tensor(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1),
        device=device,
        dtype=torch.float32,
    )
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)

    return means, colors, opacity, scales, quats


def _roll_view_world2cam(V: torch.Tensor, deg: float) -> torch.Tensor:
    """Roll камеры (world->cam матрица) вокруг camera forward (оси Z в camera space)."""
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    R = torch.tensor(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=V.dtype,
        device=V.device,
    )
    return R @ V  # <- слева, т.к. это поворот в camera-space для world->cam


def _ds_zup_to_renderer_yup(v: np.ndarray) -> np.ndarray:
    """[x,y,z]_DS(Z-up) -> [x,z,-y]_Yup"""
    x, y, z = v
    return np.array([x, z, -y], dtype=np.float32)


def _get_cubemap_views(center, swap_yz: bool = False):
    center = np.array(center, dtype=np.float32)

    faces_config = {
        "front": {"target": [0, 0, -1], "up": [0, 1, 0]},
        "right": {"target": [-1, 0, 0], "up": [0, 1, 0]},
        "back": {"target": [0, 0, 1], "up": [0, 1, 0]},
        "left": {"target": [1, 0, 0], "up": [0, 1, 0]},
        "top": {"target": [0, 1, 0], "up": [0, 0, 1]},
        "bottom": {"target": [0, -1, 0], "up": [0, 0, -1]},
    }

    # Если swap_yz=True, мы хотим трактовать вход как Z-up (DreamScene360)
    # и получить матрицы под Y-up рендерер.
    if swap_yz:
        tmp = dict(faces_config)
        faces_config = {
            "front": tmp["right"],
            "right": tmp["back"],
            "back": tmp["left"],
            "left": tmp["front"],
            "top": tmp["top"],
            "bottom": tmp["bottom"],
        }
        center_use = _ds_zup_to_renderer_yup(center)
    else:
        center_use = center

    viewmats = {}
    for name, cfg in faces_config.items():
        if swap_yz:
            target = center_use + _ds_zup_to_renderer_yup(cfg["target"])
            up = _ds_zup_to_renderer_yup(cfg["up"])
        else:
            target = center_use + np.array(cfg["target"])
            up = np.array(cfg["up"])

        z = (center_use - target) / np.linalg.norm(center_use - target)
        x = np.cross(up, z) / np.linalg.norm(np.cross(up, z))
        y = np.cross(z, x)

        R = np.stack([x, y, z], axis=0)
        t = -R @ center_use

        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = t
        viewmats[name] = torch.tensor(mat, dtype=torch.float32)

    if swap_yz:
        # После Z-up -> Y-up обычно требуется довернуть top/bottom по roll.
        # Если вдруг стало "в другую сторону", просто поменяйте знаки:
        viewmats["top"] = _roll_view_world2cam(viewmats["top"], 90.0)
        viewmats["bottom"] = _roll_view_world2cam(viewmats["bottom"], -90.0)

    return viewmats


def render_cubemap(
    ply_path,
    center_pos,
    size,
    output_path: str,
    render_depth: bool = False,
    swap_yz: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    means, colors, opacities, scales, quats = _load_gs_ply(ply_path, device)
    view_configs = _get_cubemap_views(center_pos, swap_yz=swap_yz)

    focal = size / 2.0
    K = torch.tensor(
        [[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    results = {}
    depth_results = {}

    render_mode = "RGB+ED" if render_depth else "RGB"

    for name, viewmat in view_configs.items():
        viewmat_input = viewmat.to(device).unsqueeze(0)

        with torch.no_grad():
            render_output, _, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmat_input,
                K,
                size,
                size,
                near_plane=0.01,
                far_plane=1000.0,
                render_mode=render_mode,
            )

        if render_mode == "RGB+ED":
            render_colors = render_output[0, :, :, :3]
            depth_map = render_output[0, :, :, 3].cpu().numpy()
            depth_results[name] = depth_map
        else:
            render_colors = render_output[0]

        img = (render_colors.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        output_path_pieces = output_path.split(".")
        view_name = output_path_pieces[0] + "_" + name + "." + output_path_pieces[-1]
        cv2.imwrite(view_name, img_bgr)
        results[name] = img_bgr

    if render_depth:
        results["depth_faces"] = depth_results

    return results


def _faces_to_pano(
    faces: dict, output_path: str, width: int = 4096, is_depth: bool = False
):
    faces_view = [
        faces["front"],
        faces["right"],
        faces["back"],
        faces["left"],
        faces["top"],
        faces["bottom"],
    ]

    height = width // 2
    if is_depth:
        # For depth maps, use float32 conversion
        equi_img = py360convert.c2e(
            faces_view, height, width, cube_format="list", mode="bilinear"
        )
        # Save as EXR or PNG based on extension
        if output_path.lower().endswith(".exr"):
            _save_depth_exr(equi_img.astype(np.float32), output_path)
        elif output_path.lower().endswith(".npy"):
            np.save(output_path, equi_img.astype(np.float32))
        else:
            # Save as 16-bit PNG in millimeters
            depth_mm = np.clip(equi_img * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(output_path, depth_mm)
    else:
        # For RGB images
        equi_img = py360convert.c2e(faces_view, height, width, cube_format="list")
        cv2.imwrite(output_path, equi_img)


def render_pano(
    ply_path,
    center_pos=[0, 0, 0],
    size=1024,
    output_path="pano_render.png",
    output_depth_path=None,
    swap_yz: bool = False,
):
    render_depth = output_depth_path is not None
    result = render_cubemap(
        ply_path,
        center_pos,
        size,
        output_path,
        render_depth=render_depth,
        swap_yz=swap_yz,
    )

    # Convert RGB faces to panorama
    rgb_faces = {k: v for k, v in result.items() if k != "depth_faces"}
    _faces_to_pano(rgb_faces, output_path, size, is_depth=False)

    if render_depth and "depth_faces" in result:
        _faces_to_pano(result["depth_faces"], output_depth_path, size, is_depth=True)

    return 


def _Rz(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _Rzup_to_yup_basis() -> np.ndarray:
    return np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float64)


def _correct_camera_orientation(
    pano_location, camera_location, pano_rt, camera_rt, swap_yz: bool = False
):
    """
    Corrects the camera position from global coordinates for a centered Gaussian Splatting scene
    """
    pano_R = pano_rt[:, :3]
    camera_R = camera_rt[:, :3]

    # Camera position in scene
    world_vector = camera_location - pano_location
    camera_position = pano_R @ world_vector

    R_view = camera_R @ pano_R.T
    t_view = -R_view @ camera_position

    viewmat = np.eye(4)
    viewmat[:3, :3] = R_view
    viewmat[:3, 3] = t_view
    if swap_yz:
        viewmat[:3, :3] = viewmat[:3, :3] @ _Rzup_to_yup_basis() @ _Rz(-90)
    return viewmat


def render_camera(
    ply_path,
    pano_location,
    camera_location,
    pano_rt,
    camera_rt,
    camera_k,
    output_path="result_view.png",
    output_depth_path: str = None,
    swap_yz: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    viewmat = _correct_camera_orientation(
        pano_location, camera_location, pano_rt, camera_rt, swap_yz=swap_yz
    )
    K_raw = np.array(camera_k)
    W, H = int(K_raw[0, 2] * 2), int(K_raw[1, 2] * 2)

    viewmat_tensor = torch.tensor(
        viewmat, dtype=torch.float32, device=device
    ).unsqueeze(0)
    K_tensor = torch.tensor(K_raw, dtype=torch.float32, device=device).unsqueeze(0)

    means, colors, opacities, scales, quats = _load_gs_ply(ply_path, device)

    # RGB+ED (Expected Depth) is preferred over RGB+D for depth evaluation
    # as it accounts for Gaussian transparency and better matches ground truth
    render_mode = "RGB+ED" if output_depth_path is not None else "RGB"

    with torch.no_grad():
        render_output, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmat_tensor,
            K_tensor,
            W,
            H,
            near_plane=0.01,
            far_plane=1000.0,
            render_mode=render_mode,
        )

    if render_mode == "RGB+ED":
        render_colors = render_output[0, :, :, :3]
        depth_map_tensor = render_output[0, :, :, 3]
    else:
        render_colors = render_output[0]
        depth_map_tensor = None

    img = (render_colors.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    depth_map = None
    if output_depth_path is not None and depth_map_tensor is not None:
        depth_map = depth_map_tensor.cpu().numpy()
        if output_depth_path.lower().endswith(".exr"):
            _save_depth_exr(depth_map, output_depth_path)
        elif output_depth_path.lower().endswith(".npy"):
            np.save(output_depth_path, depth_map)
        else:
            depth_map_mm = np.clip(depth_map * 1000.0, 0, 65535)
            depth_map_scaled = depth_map_mm.astype(np.uint16)
            cv2.imwrite(output_depth_path, depth_map_scaled)

    return img, depth_map


def render_virtual_camera(
    ply_path: str,
    camera: Dict,
    output_path: str,
    output_depth_path: Optional[str] = None,
    center_pos: Sequence[float] = (0.0, 0.0, 0.0),
    swap_yz: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Render a Gaussian Splatting scene from a virtual camera defined by yaw / pitch / FOV.

    The camera dict format matches the output of
    :func:`utils.pano_virtual_cameras.get_virtual_cameras`.  In particular,
    ``camera["R"]`` is the world-to-camera rotation matrix (rows = right, up, back)
    and ``camera["K"]`` is the 3×3 pinhole intrinsic matrix.

    Args:
        ply_path:          Path to the Gaussian Splatting PLY file.
        camera:            Camera dict with keys ``R``, ``K``, ``width``, ``height``.
        output_path:       File path where the RGB render is saved (PNG recommended).
        output_depth_path: Optional file path for the depth map.
                           Extension determines format: ``.exr`` → OpenEXR float32,
                           ``.npy`` → NumPy array, otherwise 16-bit PNG (mm).
        center_pos:        Camera position in world space.  For scenes generated with
                           the default WorldGen pipeline the camera sits at the origin
                           ``(0, 0, 0)``.

    Returns:
        Tuple ``(rgb_image, depth_map)``.
        ``rgb_image`` is uint8 BGR (H, W, 3) matching OpenCV convention.
        ``depth_map`` is float32 (H, W) in metres, or ``None`` when depth is not
        requested.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R = np.asarray(camera["R"], dtype=np.float64)  # (3, 3) world-to-camera rotation
    K = np.asarray(camera["K"], dtype=np.float32)  # (3, 3) intrinsics
    W, H = int(camera["width"]), int(camera["height"])

    center = np.asarray(center_pos, dtype=np.float64)
    if swap_yz:
        S = _Rzup_to_yup_basis() @ _Rz(-90.0)   # тот же “мировой” фикс, что у вас в основном рендере

        # 1) меняем базис у world->cam (справа)
        R = R @ S

        # 2) переводим center в новый базис (inverse = transpose)
        center = (S.T @ center)

    t = (-R @ center).astype(np.float32)

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R.astype(np.float32)
    viewmat[:3, 3] = t

    viewmat_tensor = torch.tensor(
        viewmat, dtype=torch.float32, device=device
    ).unsqueeze(0)
    K_tensor = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)

    means, colors, opacities, scales, quats = _load_gs_ply(ply_path, device)

    render_mode = "RGB+ED" if output_depth_path is not None else "RGB"

    with torch.no_grad():
        render_output, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmat_tensor,
            K_tensor,
            W,
            H,
            near_plane=0.01,
            far_plane=1000.0,
            render_mode=render_mode,
        )

    if render_mode == "RGB+ED":
        render_colors = render_output[0, :, :, :3]
        depth_map = render_output[0, :, :, 3].cpu().numpy()
    else:
        render_colors = render_output[0]
        depth_map = None

    img = (render_colors.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if output_depth_path is not None and depth_map is not None:
        depth_dir = os.path.dirname(output_depth_path)
        if depth_dir:
            os.makedirs(depth_dir, exist_ok=True)
        if output_depth_path.lower().endswith(".exr"):
            _save_depth_exr(depth_map, output_depth_path)
        elif output_depth_path.lower().endswith(".npy"):
            np.save(output_depth_path, depth_map)
        else:
            depth_mm = np.clip(depth_map * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(output_depth_path, depth_mm)

    return img, depth_map
