import torch
import numpy as np
import cv2
import os
from gsplat.rendering import rasterization
import py360convert
from gs_utils import load_gs_ply


def get_cubemap_views(center):
    """Определяет параметры для 6 граней куба"""
    center = np.array(center)
    # Направления взгляда и векторы "вверх"
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


def render_cubemap(ply_path, center_pos=[0, 0, 0], size=524, output_dir="cubemap_out"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Загрузка модели
    means, colors, opacities, scales, quats = load_gs_ply(ply_path, device)
    view_configs = get_cubemap_views(center_pos)

    # 2. Общие интринсики (FOV 90 градусов)
    focal = size / 2.0
    K = torch.tensor(
        [[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(
        0
    )  # (1, 3, 3)

    # 3. Рендеринг в цикле (по одной грани за раз)
    print(f"Rendering cubemap faces for {means.shape[0]} gaussians...")

    results = {}
    for name, viewmat in view_configs.items():
        print(f" -> Rendering {name}...")
        print(viewmat)
        # Добавляем размерность батча (1) для gsplat
        viewmat_input = viewmat.to(device).unsqueeze(0)  # (1, 4, 4)

        with torch.no_grad():
            # gsplat.rasterization ожидает тензоры без батч-размерности для данных
            # или с батч-размерностью, если она совпадает с viewmats.
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

        # Конвертация и сохранение
        img = (render_colors[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), img_bgr)
        results[name] = img_bgr

    print(f"All faces saved in '{output_dir}/'")
    return results


def create_pano(faces: dict, output_dir="cubemap_out"):
    faces_view = [
        faces["front"],
        faces["right"],
        faces["back"],
        faces["left"],
        faces["top"],
        faces["bottom"],
    ]
    equi_img = py360convert.c2e(faces_view, 2048, 4096, cube_format="list")
    pano_path = os.path.join(output_dir, f"pano_render.png")
    cv2.imwrite(pano_path, equi_img)


# --- ПАРАМЕТРЫ И ЗАПУСК ---
ply_file = "/mnt/e/3D/experiments/output/WorldGen/splat.ply"
# Координаты из вашего JSON
cam_loc = [0, 0, 0]

faces = render_cubemap(ply_file, center_pos=cam_loc, size=1024)
create_pano(faces)
