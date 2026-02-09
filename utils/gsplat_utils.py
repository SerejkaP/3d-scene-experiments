import torch
import numpy as np
from plyfile import PlyData
import cv2
import os
from gsplat.rendering import rasterization
import py360convert
import OpenEXR
import Imath


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


def _get_cubemap_views(center):
    center = np.array(center)
    faces_config = {
        "front": {"target": [0, 0, -1], "up": [0, 1, 0]},
        "right": {"target": [-1, 0, 0], "up": [0, 1, 0]},
        "back": {"target": [0, 0, 1], "up": [0, 1, 0]},
        "left": {"target": [1, 0, 0], "up": [0, 1, 0]},
        "top": {"target": [0, 1, 0], "up": [0, 0, 1]},
        "bottom": {"target": [0, -1, 0], "up": [0, 0, -1]},
    }

    viewmats = {}
    for name, cfg in faces_config.items():
        target = center + np.array(cfg["target"])
        up = np.array(cfg["up"])

        z = (center - target) / np.linalg.norm(center - target)
        x = np.cross(up, z) / np.linalg.norm(np.cross(up, z))
        y = np.cross(z, x)

        R = np.stack([x, y, z], axis=0)
        t = -R @ center

        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = t
        viewmats[name] = torch.tensor(mat, dtype=torch.float32)

    return viewmats


def render_cubemap(
    ply_path, center_pos, size, output_path: str, render_depth: bool = False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    means, colors, opacities, scales, quats = _load_gs_ply(ply_path, device)
    view_configs = _get_cubemap_views(center_pos)

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
):
    render_depth = output_depth_path is not None
    result = render_cubemap(
        ply_path, center_pos, size, output_path, render_depth=render_depth
    )

    # Convert RGB faces to panorama
    rgb_faces = {k: v for k, v in result.items() if k != "depth_faces"}
    _faces_to_pano(rgb_faces, output_path, size, is_depth=False)

    # Convert depth faces to panorama if requested
    depth_pano = None
    if render_depth and "depth_faces" in result:
        _faces_to_pano(result["depth_faces"], output_depth_path, size, is_depth=True)

        # Load the saved depth for return
        depth_pano = load_depth(output_depth_path, depth_scale=1.0)

    return depth_pano


def _correct_camera_orientation(pano_location, camera_location, pano_rt, camera_rt):
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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    viewmat = _correct_camera_orientation(
        pano_location, camera_location, pano_rt, camera_rt
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
