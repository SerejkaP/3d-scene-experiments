import torch
import numpy as np
from plyfile import PlyData
import cv2
import os
from gsplat.rendering import rasterization
import py360convert


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


def render_cubemap(ply_path, center_pos, size, output_path: str):
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
    for name, viewmat in view_configs.items():
        viewmat_input = viewmat.to(device).unsqueeze(0)

        with torch.no_grad():
            render_colors, _, _ = rasterization(
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
            )
        img = (render_colors[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        output_path_pieces = output_path.split(".")
        view_name = output_path_pieces[0] + "_" + name + "." + output_path_pieces[-1]
        cv2.imwrite(view_name, img_bgr)
        results[name] = img_bgr
    return results


def _faces_to_pano(faces: dict, output_path):
    faces_view = [
        faces["front"],
        faces["right"],
        faces["back"],
        faces["left"],
        faces["top"],
        faces["bottom"],
    ]
    equi_img = py360convert.c2e(faces_view, 2048, 4096, cube_format="list")
    cv2.imwrite(output_path, equi_img)


def render_pano(
    ply_path, center_pos=[0, 0, 0], size=1024, output_path="pano_render.png"
):
    faces = render_cubemap(ply_path, center_pos, size, output_path)
    _faces_to_pano(faces, output_path)


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

    viewmat_tensor = torch.tensor(
        viewmat, dtype=torch.float32, device=device
    ).unsqueeze(0)
    K_tensor = torch.tensor(K_raw, dtype=torch.float32, device=device).unsqueeze(0)

    means, colors, opacities, scales, quats = _load_gs_ply(ply_path, device)
    with torch.no_grad():
        render_colors, _, _ = rasterization(
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
        )

    img = (render_colors[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
