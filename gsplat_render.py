import torch
import numpy as np
import json
import cv2
from gsplat.rendering import rasterization
from utils.gs_utils import load_gs_ply


def correct_camera_orientation(pano_location, camera_location, pano_rt, camera_rt):
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

    viewmat = correct_camera_orientation(
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

    means, colors, opacities, scales, quats = load_gs_ply(ply_path, device)
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
    print(f"Image {W}x{H} saved to {output_path}")


if __name__ == "__main__":
    # Запуск
    ply_file = "/mnt/e/3D/experiments/output/WorldGen/splat.ply"
    pano_json_path = "/mnt/d/datasets/2D-3D-Semantics/area_1/pano/pose/camera_00d10d86db1e435081a837ced388375f_office_24_frame_equirectangular_domain_pose.json"
    camera_json_path = "/mnt/d/datasets/2D-3D-Semantics/area_1/data/pose/camera_00d10d86db1e435081a837ced388375f_office_24_frame_39_domain_pose.json"

    dataset = "2D-3D-Semantic"

    with open(pano_json_path, "r") as f:
        pano_data = json.load(f)

    with open(camera_json_path, "r") as f:
        camera_data = json.load(f)

    render_camera(
        ply_file,
        pano_location=np.array(pano_data["camera_location"]),
        camera_location=np.array(camera_data["camera_location"]),
        pano_rt=np.array(pano_data["camera_rt_matrix"]),
        camera_rt=np.array(camera_data["camera_rt_matrix"]),
        camera_k=np.array(camera_data["camera_k_matrix"]),
        output_path="final_gs_render39.png",
    )
