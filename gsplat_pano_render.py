import torch
import numpy as np
import cv2
import os
from gsplat.rendering import rasterization
import py360convert
from utils.gs_utils import load_gs_ply


def get_cubemap_views(center):
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

    means, colors, opacities, scales, quats = load_gs_ply(ply_path, device)
    view_configs = get_cubemap_views(center_pos)

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
        print(view_name)
        cv2.imwrite(view_name, img_bgr)
        results[name] = img_bgr

    print(f"All faces saved in '{output_dir}/'")
    return results


def faces_to_pano(faces: dict, output_path):
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
    faces_to_pano(faces, output_path)


if __name__ == "__main__":
    ply_file = "/mnt/e/3D/experiments/output/WorldGen/splat.ply"
    cam_loc = [0, 0, 0]

    render_pano(ply_file, cam_loc, 1024, "output/pano_render.png")
