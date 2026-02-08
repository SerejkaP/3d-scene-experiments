import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import json
from tqdm import tqdm
from utils.gsplat_utils import render_camera, render_pano, load_depth
from utils.worldgen_utils import worldgen_generate
from utils.dataset_2d3ds_utils import pose_json_by_image_path
from utils.metrics import (
    ClipDistanceMetric,
    compute_gt_metrics,
    LpipsMetric,
    compute_brisque,
    compute_niqe,
    FidMetric,
    DepthMetrics,
)
from utils.ob3d_utils.eval_nvs import calculate_metrics
from utils.ob3d_utils.eval_depth import (
    calculate_rmse,
    calculate_abs_relative_difference,
)
from utils.tensorboard_logger import TensorBoardLogger


def create_gs(model_name, pano_path, save_path):
    if model_name == "worldgen":
        return worldgen_generate(pano_path, save_path)
    else:
        raise Exception("Undefined model name!")


def render_2d3ds(ply_file, pano_json_path, camera_json_path, output_path):
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
        output_path=output_path,
    )


def render_ob3d(ply_file, camera_pose_path, output_path, output_depth_path=None):
    with open(camera_pose_path, "r") as f:
        data = json.load(f)
        data = data[0]

    rotation_w2c = np.array(data["extrinsics"]["rotation"])  # w2c
    translation_w2c = np.array(data["extrinsics"]["translation"])  # w2c
    rotation_c2w = rotation_w2c.T
    translation_c2w = -rotation_c2w @ translation_w2c
    se3 = np.eye(4)
    se3[:3, :3] = rotation_c2w
    se3[:3, 3] = translation_c2w


def render_structured3d(
    ply_file, pano_position, camera_pose_path, output_path, output_depth_path=None
):
    """
    Render perspective view for Structured3D dataset.

    Args:
        ply_file: Path to PLY file with Gaussian Splatting scene
        pano_position: Panorama position in Structured3D coordinates (millimeters)
        camera_pose_path: Path to camera_pose.txt file
        output_path: Path to save rendered RGB image
        output_depth_path: Optional path to save rendered depth map

    Returns:
        Tuple of (rgb_image, depth_map) as numpy arrays
    """
    values = np.loadtxt(camera_pose_path)
    camera_location = values[0:3]

    # Structured3D camera_pose.txt: position[3] forward[3] up[3] xfov yfov
    forward = values[3:6]
    up = values[6:9]
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)

    xfov, yfov = values[9], values[10]
    W, H = 1280, 720
    fx = (W / 2) / np.tan(xfov)
    fy = (H / 2) / np.tan(yfov)
    camera_k = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])

    # Structured3D uses millimeters with Z-up coordinate system.
    # WorldGen GS scene uses meters with OpenCV convention (+X right, +Y down, +Z forward).
    pano_location_m = pano_position / 1000.0
    camera_location_m = camera_location / 1000.0

    # OpenCV world-to-camera rotation: rows are [right, down, forward].
    camera_R_cv = np.stack([right, down, forward], axis=0)

    # Rotation from Structured3D world (Z-up) to GS scene (OpenCV camera space).
    # Panorama center faces +Y direction in Structured3D world.
    # S3D +X (right) -> GS +X, S3D +Z (up) -> GS -Y, S3D +Y (forward) -> GS +Z.
    pano_rt = np.array(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ],
        dtype=np.float64,
    )

    return render_camera(
        ply_file,
        pano_location=pano_location_m,
        camera_location=camera_location_m,
        pano_rt=pano_rt,
        camera_rt=camera_R_cv,
        camera_k=camera_k,
        output_path=output_path,
        output_depth_path=output_depth_path,
    )


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    save_path = os.path.join(cfg.save_path, cfg.model.name, cfg.dataset.name)
    os.makedirs(save_path, exist_ok=True)
    if cfg.dataset.name == "2d3ds":
        pano_path = os.path.join(cfg.dataset.path, "pano")
        pano_pose = os.path.join(pano_path, "pose")
        pano_images = os.path.join(pano_path, "rgb")

        data_path = os.path.join(cfg.dataset.path, "data")
        data_pose = os.path.join(data_path, "pose")
        data_images = os.path.join(data_path, "rgb")
        pano_img_list = [
            img_name
            for img_name in os.listdir(pano_images)
            if img_name.endswith(".png")
        ]
        if len(pano_img_list) == 0:
            raise Exception("No images in directory!!!")

        current_pano_rgb_name = pano_img_list[0]

        pano_json_path, pano_name = pose_json_by_image_path(
            current_pano_rgb_name, pano_pose
        )

        tb_logger = TensorBoardLogger(
            log_dir=cfg.tensorboard.log_dir or save_path,
            enabled=cfg.tensorboard.enabled,
            flush_secs=cfg.tensorboard.flush_secs,
            max_image_size=2048,
        )
        lpips_metric = LpipsMetric()
        clip_metric = ClipDistanceMetric()
        fid_metric = FidMetric()

        num_iters = min(cfg.generation_iters, len(pano_img_list))
        for i in range(num_iters):
            current_pano_rgb_path = os.path.join(pano_images, current_pano_rgb_name)
            rendered_pano_path = os.path.join(save_path, f"{pano_name}_pano.png")
            ply_render_path = os.path.join(save_path, f"{pano_name}_render.ply")

            if cfg.generate:
                print(f"Generate scene for {current_pano_rgb_name}")
                # Generation time metric
                generation_time = create_gs(
                    cfg.model.name, current_pano_rgb_path, ply_render_path
                )
                tb_logger.log_scalar(
                    "Performance/generation_time_seconds", generation_time, i
                )
                print("Generation time: ", generation_time)

                render_pano(
                    ply_render_path,
                    [0, 0, 0],
                    cfg.dataset.pano_width,
                    rendered_pano_path,
                )

            # Panorama metrics (reference-based)
            pano_lpips = lpips_metric.compute(current_pano_rgb_path, rendered_pano_path)
            pano_clip_d = clip_metric.compute(current_pano_rgb_path, rendered_pano_path)
            pano_psnr, pano_ssim = compute_gt_metrics(
                current_pano_rgb_path, rendered_pano_path
            )
            # Panorama metrics (reference-free)
            pano_niqe = compute_niqe(rendered_pano_path)
            pano_brisque = compute_brisque(rendered_pano_path)

            tb_logger.log_scalar(f"Metrics/Panorama/LPIPS", pano_lpips, i)
            tb_logger.log_scalar(f"Metrics/Panorama/CLIP-Distance", pano_clip_d, i)
            tb_logger.log_scalar(f"Metrics/Panorama/PSNR", pano_psnr, i)
            tb_logger.log_scalar(f"Metrics/Panorama/SSIM", pano_ssim, i)
            tb_logger.log_scalar(f"Metrics/Panorama/NIQE", pano_niqe, i)
            tb_logger.log_scalar(f"Metrics/Panorama/BRISQUE", pano_brisque, i)

            print(f"PANO LPIPS: {pano_lpips}")
            print(f"PANO CLIP-Distance: {pano_clip_d}")
            print(f"PANO PSNR: {pano_psnr}")
            print(f"PANO SSIM: {pano_ssim}")
            print(f"PANO NIQE: {pano_niqe}")
            print(f"PANO BRISQUE: {pano_brisque}")

            if cfg.tensorboard.log_images:
                tb_logger.log_image_comparison(
                    f"Images/Panorama/{pano_name}",
                    current_pano_rgb_path,
                    rendered_pano_path,
                    i,
                )

            camera_poses = [
                camera_json
                for camera_json in os.listdir(data_pose)
                if camera_json.startswith(f"{pano_name}_")
            ]

            total_lpips = 0
            total_clip_distance = 0
            total_psnr = 0
            total_ssim = 0
            total_niqe = 0
            total_brisque = 0
            gt_paths_for_fid = []
            render_paths_for_fid = []
            for camera_idx, camera_pose in enumerate(tqdm(camera_poses)):
                camera_json_path = os.path.join(data_pose, camera_pose)
                rendered_camera_subname = "_".join(camera_pose.split("_")[:-1])
                rendered_camera = f"{rendered_camera_subname}_render.png"
                rendered_camera_path = os.path.join(save_path, rendered_camera)

                render_2d3ds(
                    ply_render_path,
                    pano_json_path,
                    camera_json_path,
                    rendered_camera_path,
                )

                gt_image_path = os.path.join(
                    data_images, f"{rendered_camera_subname}_rgb.png"
                )

                # Camera metrics (reference-based)
                lpips = lpips_metric.compute(gt_image_path, rendered_camera_path)
                clip_distance = clip_metric.compute(gt_image_path, rendered_camera_path)
                psnr, ssim = compute_gt_metrics(gt_image_path, rendered_camera_path)
                # Camera metrics (reference-free)
                niqe = compute_niqe(rendered_camera_path)
                brisque = compute_brisque(rendered_camera_path)

                tb_logger.log_scalar(
                    f"Metrics/Camera_{pano_name}/LPIPS", lpips, camera_idx
                )
                tb_logger.log_scalar(
                    f"Metrics/Camera_{pano_name}/CLIP-Distance",
                    clip_distance,
                    camera_idx,
                )
                tb_logger.log_scalar(
                    f"Metrics/Camera_{pano_name}/PSNR", psnr, camera_idx
                )
                tb_logger.log_scalar(
                    f"Metrics/Camera_{pano_name}/SSIM", ssim, camera_idx
                )
                tb_logger.log_scalar(
                    f"Metrics/Camera_{pano_name}/NIQE", niqe, camera_idx
                )
                tb_logger.log_scalar(
                    f"Metrics/Camera_{pano_name}/BRISQUE", brisque, camera_idx
                )

                if (
                    cfg.tensorboard.log_images
                    and camera_idx < cfg.tensorboard.max_images_per_scene
                ):
                    tb_logger.log_image_comparison(
                        f"Images/Camera/{rendered_camera_subname}",
                        gt_image_path,
                        rendered_camera_path,
                        i,
                    )

                total_lpips += lpips
                total_clip_distance += clip_distance
                total_psnr += psnr
                total_ssim += ssim
                total_niqe += niqe
                total_brisque += brisque
                gt_paths_for_fid.append(gt_image_path)
                render_paths_for_fid.append(rendered_camera_path)

            scene_lpips = total_lpips / len(camera_poses)
            scene_clip_d = total_clip_distance / len(camera_poses)
            scene_psnr = total_psnr / len(camera_poses)
            scene_ssim = total_ssim / len(camera_poses)
            scene_niqe = total_niqe / len(camera_poses)
            scene_brisque = total_brisque / len(camera_poses)
            scene_fid = fid_metric.compute(gt_paths_for_fid, render_paths_for_fid)

            tb_logger.log_scalar("Metrics/Scene/LPIPS_mean", scene_lpips, i)
            tb_logger.log_scalar("Metrics/Scene/CLIP-Distance_mean", scene_clip_d, i)
            tb_logger.log_scalar("Metrics/Scene/PSNR_mean", scene_psnr, i)
            tb_logger.log_scalar("Metrics/Scene/SSIM_mean", scene_ssim, i)
            tb_logger.log_scalar("Metrics/Scene/NIQE_mean", scene_niqe, i)
            tb_logger.log_scalar("Metrics/Scene/BRISQUE_mean", scene_brisque, i)
            tb_logger.log_scalar("Metrics/Scene/FID", scene_fid, i)

            print(f"Scene average LPIPS: {scene_lpips}")
            print(f"Scene average CLIP-Distance: {scene_clip_d}")
            print(f"Scene average PSNR: {scene_psnr}")
            print(f"Scene average SSIM: {scene_ssim}")
            print(f"Scene average NIQE: {scene_niqe}")
            print(f"Scene average BRISQUE: {scene_brisque}")
            print(f"Scene FID: {scene_fid}")

            current_pano_rgb_name = pano_img_list[i]
            pano_json_path, pano_name = pose_json_by_image_path(
                current_pano_rgb_name, pano_pose
            )
        tb_logger.close()

    elif cfg.dataset.name == "ob3d":
        tb_logger = TensorBoardLogger(
            log_dir=cfg.tensorboard.log_dir or save_path,
            enabled=cfg.tensorboard.enabled,
            flush_secs=cfg.tensorboard.flush_secs,
            max_image_size=2048,
        )
        lpips_metric = LpipsMetric()
        clip_metric = ClipDistanceMetric()
        fid_metric = FidMetric()
        counter = 0
        dataset_path = str(cfg.dataset.path)
        for scene in os.listdir(dataset_path):
            if counter >= cfg.generation_iters:
                break
            scene_dict = {
                "barbershop": 10,
                "archiviz-flat": 20,
                "bistro": 50,
                "classroom": 15,
                "emerald-square": 70,
                "fisher-hut": 70,
                "lone-monk": 25,
                "restroom": 20,
                "san-miguel": 20,
                "sponza": 20,
                "sun_temple": 20,
                "pavillion": 50,
            }
            depth_metric = DepthMetrics(min_depth=0.0, max_depth=scene_dict[scene])
            for scene_type in ["Egocentric", "Non-Egocentric"]:
                if counter >= cfg.generation_iters:
                    break
                # cameras/, depth/, images/, normals/, sparse/, test.txt, train.txt
                scene_path = os.path.join(dataset_path, scene, scene_type)
                images_path = os.path.join(scene_path, "images")
                depth_path = os.path.join(scene_path, "depths")
                test_data = []
                with open(os.path.join(scene_path, "test.txt"), "r") as f:
                    test_data = [int(line.strip()) for line in f.readlines()]
                for t in test_data:
                    if counter >= cfg.generation_iters:
                        break
                    t_scene = f"{str(t).zfill(5)}"
                    rendered_pano_path = os.path.join(
                        save_path, scene, scene_type, f"{t_scene}_pano.png"
                    )
                    rendered_pano_depth_path = os.path.join(
                        save_path, scene, scene_type, f"{t_scene}_depth.exr"
                    )
                    ply_render_path = os.path.join(
                        save_path, f"{scene}_{scene_type}_{t_scene}_render.ply"
                    )
                    pano_rgb_path = os.path.join(images_path, f"{t_scene}_rgb.png")
                    gt_pano_depth_path = os.path.join(
                        depth_path, f"{t_scene}_depth.exr"
                    )
                    room_name = f"{scene}/{scene_type}/{t_scene}"
                    if cfg.generate:
                        print(f"Generate scene for {room_name}_rgb.png")
                        # Generation time metric
                        generation_time = create_gs(
                            cfg.model.name, pano_rgb_path, ply_render_path
                        )
                        tb_logger.log_scalar(
                            "Performance/generation_time_seconds",
                            generation_time,
                            counter,
                        )
                        print("Generation time: ", generation_time)

                        render_pano(
                            ply_render_path,
                            [0, 0, 0],
                            cfg.dataset.pano_width,
                            rendered_pano_path,
                            rendered_pano_depth_path,
                        )

                    # Panorama metrics (reference-based)
                    pano_lpips = lpips_metric.compute(pano_rgb_path, rendered_pano_path)
                    pano_clip_d = clip_metric.compute(pano_rgb_path, rendered_pano_path)
                    pano_psnr, pano_ssim = compute_gt_metrics(
                        pano_rgb_path, rendered_pano_path
                    )
                    # Panorama metrics (reference-free)
                    pano_niqe = compute_niqe(rendered_pano_path)
                    pano_brisque = compute_brisque(rendered_pano_path)

                    # Panorama depth metrics
                    pano_depth_rmse = float("nan")
                    pano_depth_abs_rel = float("nan")

                    if os.path.exists(gt_pano_depth_path) and os.path.exists(
                        rendered_pano_depth_path
                    ):
                        try:
                            # Load panorama depth maps in EXR (meters)
                            gt_pano_depth = load_depth(gt_pano_depth_path)
                            # Rendered panorama depth is in EXR (meters)
                            render_pano_depth = load_depth(
                                rendered_pano_depth_path,
                                depth_scale=(
                                    1.0 / 1000.0
                                    if rendered_pano_depth_path.endswith(".png")
                                    else 1.0
                                ),
                            )

                            pano_depth_metrics = depth_metric.compute(
                                gt_pano_depth, render_pano_depth
                            )

                            if not np.isnan(pano_depth_metrics["rmse"]):
                                pano_depth_rmse = pano_depth_metrics["rmse"]
                                pano_depth_abs_rel = pano_depth_metrics["abs_rel"]

                                tb_logger.log_scalar(
                                    "Metrics/Panorama/Depth_RMSE",
                                    pano_depth_rmse,
                                    counter,
                                )
                                tb_logger.log_scalar(
                                    "Metrics/Panorama/Depth_AbsRel",
                                    pano_depth_abs_rel,
                                    counter,
                                )
                        except Exception as e:
                            print(
                                f"Warning: Could not compute panorama depth metrics for {room_name}: {e}"
                            )

                    tb_logger.log_scalar("Metrics/Panorama/LPIPS", pano_lpips, counter)
                    tb_logger.log_scalar(
                        "Metrics/Panorama/CLIP-Distance", pano_clip_d, counter
                    )
                    tb_logger.log_scalar("Metrics/Panorama/PSNR", pano_psnr, counter)
                    tb_logger.log_scalar("Metrics/Panorama/SSIM", pano_ssim, counter)
                    tb_logger.log_scalar("Metrics/Panorama/NIQE", pano_niqe, counter)
                    tb_logger.log_scalar(
                        "Metrics/Panorama/BRISQUE", pano_brisque, counter
                    )

                    print(f"PANO LPIPS: {pano_lpips}")
                    print(f"PANO CLIP-Distance: {pano_clip_d}")
                    print(f"PANO PSNR: {pano_psnr}")
                    print(f"PANO SSIM: {pano_ssim}")
                    print(f"PANO NIQE: {pano_niqe}")
                    print(f"PANO BRISQUE: {pano_brisque}")
                    if not np.isnan(pano_depth_rmse):
                        print(f"PANO Depth RMSE: {pano_depth_rmse:.4f}")
                        print(f"PANO Depth Abs Rel: {pano_depth_abs_rel:.4f}")

                    if cfg.tensorboard.log_images:
                        tb_logger.log_image_comparison(
                            f"Images/Panorama/{room_name}",
                            pano_rgb_path,
                            rendered_pano_path,
                            counter,
                        )

                    counter += 1
        tb_logger.close()

    elif cfg.dataset.name == "structured3d":
        dataset_path = str(cfg.dataset.path)
        scenes = sorted([s for s in os.listdir(dataset_path) if s.startswith("scene_")])

        tb_logger = TensorBoardLogger(
            log_dir=cfg.tensorboard.log_dir or save_path,
            enabled=cfg.tensorboard.enabled,
            flush_secs=cfg.tensorboard.flush_secs,
            max_image_size=2048,
        )
        lpips_metric = LpipsMetric()
        clip_metric = ClipDistanceMetric()
        fid_metric = FidMetric()
        depth_metric = DepthMetrics(min_depth=0.0, max_depth=100.0)

        room_idx = 0
        for scene in scenes:
            if room_idx >= cfg.generation_iters:
                break

            rendering_path = os.path.join(dataset_path, scene, "2D_rendering")
            if not os.path.isdir(rendering_path):
                continue

            rooms = sorted(os.listdir(rendering_path))
            for room in rooms:
                if room_idx >= cfg.generation_iters:
                    break

                room_path = os.path.join(rendering_path, room)
                panorama_dir = os.path.join(room_path, "panorama")
                perspective_dir = os.path.join(room_path, "perspective", "full")

                camera_xyz_file = os.path.join(panorama_dir, "camera_xyz.txt")
                pano_rgb_path = os.path.join(panorama_dir, "full", "rgb_rawlight.png")

                if not os.path.exists(camera_xyz_file) or not os.path.exists(
                    pano_rgb_path
                ):
                    print(f"Skipping {scene}/{room}: missing data")
                    continue

                with open(camera_xyz_file, "r") as f:
                    pano_position = np.array(
                        [float(x) for x in f.read().strip().split()]
                    )

                room_name = f"{scene}_{room}"
                room_save_path = os.path.join(save_path, scene, room)
                os.makedirs(room_save_path, exist_ok=True)

                rendered_pano_path = os.path.join(room_save_path, "pano_render.png")
                rendered_pano_depth_path = os.path.join(
                    room_save_path, "pano_depth.exr"
                )
                ply_render_path = os.path.join(room_save_path, "scene.ply")

                # Ground truth panorama depth path
                gt_pano_depth_path = os.path.join(panorama_dir, "full", "depth.png")

                if cfg.generate:
                    print(f"Generate scene for {room_name}")
                    generation_time = create_gs(
                        cfg.model.name, pano_rgb_path, ply_render_path
                    )
                    tb_logger.log_scalar(
                        "Performance/generation_time_seconds",
                        generation_time,
                        room_idx,
                    )
                    print("Generation time:", generation_time)

                    render_pano(
                        ply_render_path,
                        [0, 0, 0],
                        cfg.dataset.pano_width,
                        rendered_pano_path,
                        output_depth_path=rendered_pano_depth_path,
                    )

                # Panorama metrics (reference-based)
                pano_lpips = lpips_metric.compute(pano_rgb_path, rendered_pano_path)
                pano_clip_d = clip_metric.compute(pano_rgb_path, rendered_pano_path)
                pano_psnr, pano_ssim = compute_gt_metrics(
                    pano_rgb_path, rendered_pano_path
                )
                # Panorama metrics (reference-free)
                pano_niqe = compute_niqe(rendered_pano_path)
                pano_brisque = compute_brisque(rendered_pano_path)

                # Panorama depth metrics
                pano_depth_rmse = float("nan")
                pano_depth_abs_rel = float("nan")

                if os.path.exists(gt_pano_depth_path) and os.path.exists(
                    rendered_pano_depth_path
                ):
                    try:
                        # Load panorama depth maps
                        # Structured3D panorama depth is in millimeters (PNG uint16)
                        gt_pano_depth = load_depth(
                            gt_pano_depth_path, depth_scale=1.0 / 1000.0
                        )
                        # Rendered panorama depth is in EXR (meters)
                        render_pano_depth = load_depth(
                            rendered_pano_depth_path,
                            depth_scale=(
                                1.0 / 1000.0
                                if rendered_pano_depth_path.endswith(".png")
                                else 1.0
                            ),
                        )

                        pano_depth_metrics = depth_metric.compute(
                            gt_pano_depth, render_pano_depth
                        )

                        if not np.isnan(pano_depth_metrics["rmse"]):
                            pano_depth_rmse = pano_depth_metrics["rmse"]
                            pano_depth_abs_rel = pano_depth_metrics["abs_rel"]

                            tb_logger.log_scalar(
                                "Metrics/Panorama/Depth_RMSE", pano_depth_rmse, room_idx
                            )
                            tb_logger.log_scalar(
                                "Metrics/Panorama/Depth_AbsRel",
                                pano_depth_abs_rel,
                                room_idx,
                            )
                    except Exception as e:
                        print(
                            f"Warning: Could not compute panorama depth metrics for {room_name}: {e}"
                        )

                tb_logger.log_scalar("Metrics/Panorama/LPIPS", pano_lpips, room_idx)
                tb_logger.log_scalar(
                    "Metrics/Panorama/CLIP-Distance", pano_clip_d, room_idx
                )
                tb_logger.log_scalar("Metrics/Panorama/PSNR", pano_psnr, room_idx)
                tb_logger.log_scalar("Metrics/Panorama/SSIM", pano_ssim, room_idx)
                tb_logger.log_scalar("Metrics/Panorama/NIQE", pano_niqe, room_idx)
                tb_logger.log_scalar("Metrics/Panorama/BRISQUE", pano_brisque, room_idx)

                print(f"PANO LPIPS: {pano_lpips}")
                print(f"PANO CLIP-Distance: {pano_clip_d}")
                print(f"PANO PSNR: {pano_psnr}")
                print(f"PANO SSIM: {pano_ssim}")
                print(f"PANO NIQE: {pano_niqe}")
                print(f"PANO BRISQUE: {pano_brisque}")
                if not np.isnan(pano_depth_rmse):
                    print(f"PANO Depth RMSE: {pano_depth_rmse:.4f}")
                    print(f"PANO Depth Abs Rel: {pano_depth_abs_rel:.4f}")

                if cfg.tensorboard.log_images:
                    tb_logger.log_image_comparison(
                        f"Images/Panorama/{room_name}",
                        pano_rgb_path,
                        rendered_pano_path,
                        room_idx,
                    )

                # Perspective views
                if not os.path.isdir(perspective_dir):
                    room_idx += 1
                    continue

                view_dirs = sorted(
                    [
                        d
                        for d in os.listdir(perspective_dir)
                        if os.path.isdir(os.path.join(perspective_dir, d))
                    ]
                )

                total_lpips = 0
                total_clip_distance = 0
                total_psnr = 0
                total_ssim = 0
                all_niqe = []
                all_brisque = []
                gt_paths_for_fid = []
                render_paths_for_fid = []

                # Depth metrics accumulators
                total_rmse = 0
                total_abs_rel = 0
                depth_count = 0

                for camera_idx, view_dir in enumerate(tqdm(view_dirs)):
                    view_path = os.path.join(perspective_dir, view_dir)
                    camera_pose_file = os.path.join(view_path, "camera_pose.txt")
                    gt_image_path = os.path.join(view_path, "rgb_rawlight.png")
                    gt_depth_path = os.path.join(view_path, "depth.png")

                    if not os.path.exists(camera_pose_file) or not os.path.exists(
                        gt_image_path
                    ):
                        continue

                    rendered_camera_path = os.path.join(
                        room_save_path, f"view_{view_dir}_render.png"
                    )
                    rendered_depth_path = os.path.join(
                        room_save_path, f"view_{view_dir}_depth.exr"
                    )

                    render_structured3d(
                        ply_render_path,
                        pano_position,
                        camera_pose_file,
                        rendered_camera_path,
                        output_depth_path=rendered_depth_path,
                    )

                    # Camera metrics (reference-based)
                    lpips = lpips_metric.compute(gt_image_path, rendered_camera_path)
                    clip_distance = clip_metric.compute(
                        gt_image_path, rendered_camera_path
                    )
                    psnr, ssim = compute_gt_metrics(gt_image_path, rendered_camera_path)
                    # Camera metrics (reference-free)
                    niqe = compute_niqe(rendered_camera_path)
                    brisque = compute_brisque(rendered_camera_path)

                    # Depth metrics (if ground truth depth exists)
                    if os.path.exists(gt_depth_path) and os.path.exists(
                        rendered_depth_path
                    ):

                        gt_depth = load_depth(gt_depth_path, depth_scale=1.0 / 1000.0)
                        render_depth = load_depth(
                            rendered_depth_path,
                            depth_scale=(
                                1.0 / 1000.0
                                if rendered_depth_path.endswith(".png")
                                else 1.0
                            ),
                        )

                        depth_metrics = depth_metric.compute(gt_depth, render_depth)

                        if not np.isnan(depth_metrics["rmse"]):
                            tb_logger.log_scalar(
                                f"Metrics/Camera_{room_name}/Depth_RMSE",
                                depth_metrics["rmse"],
                                camera_idx,
                            )
                            tb_logger.log_scalar(
                                f"Metrics/Camera_{room_name}/Depth_AbsRel",
                                depth_metrics["abs_rel"],
                                camera_idx,
                            )

                            total_rmse += depth_metrics["rmse"]
                            total_abs_rel += depth_metrics["abs_rel"]
                            depth_count += 1

                    tb_logger.log_scalar(
                        f"Metrics/Camera_{room_name}/LPIPS",
                        lpips,
                        camera_idx,
                    )
                    tb_logger.log_scalar(
                        f"Metrics/Camera_{room_name}/CLIP-Distance",
                        clip_distance,
                        camera_idx,
                    )
                    tb_logger.log_scalar(
                        f"Metrics/Camera_{room_name}/PSNR",
                        psnr,
                        camera_idx,
                    )
                    tb_logger.log_scalar(
                        f"Metrics/Camera_{room_name}/SSIM",
                        ssim,
                        camera_idx,
                    )
                    tb_logger.log_scalar(
                        f"Metrics/Camera_{room_name}/NIQE",
                        niqe,
                        camera_idx,
                    )
                    tb_logger.log_scalar(
                        f"Metrics/Camera_{room_name}/BRISQUE",
                        brisque,
                        camera_idx,
                    )

                    if (
                        cfg.tensorboard.log_images
                        and camera_idx < cfg.tensorboard.max_images_per_scene
                    ):
                        tb_logger.log_image_comparison(
                            f"Images/Camera/{room_name}_{view_dir}",
                            gt_image_path,
                            rendered_camera_path,
                            room_idx,
                        )

                    total_lpips += lpips
                    total_clip_distance += clip_distance
                    total_psnr += psnr
                    total_ssim += ssim
                    all_niqe.append(niqe)
                    all_brisque.append(brisque)
                    gt_paths_for_fid.append(gt_image_path)
                    render_paths_for_fid.append(rendered_camera_path)

                num_views = len(gt_paths_for_fid)
                if num_views > 0:
                    scene_lpips = total_lpips / num_views
                    scene_clip_d = total_clip_distance / num_views
                    scene_psnr = total_psnr / num_views
                    scene_ssim = total_ssim / num_views
                    scene_niqe = float(np.nanmean(all_niqe))
                    scene_brisque = float(np.nanmean(all_brisque))
                    scene_fid = fid_metric.compute(
                        gt_paths_for_fid, render_paths_for_fid
                    )

                    tb_logger.log_scalar(
                        "Metrics/Scene/LPIPS_mean", scene_lpips, room_idx
                    )
                    tb_logger.log_scalar(
                        "Metrics/Scene/CLIP-Distance_mean",
                        scene_clip_d,
                        room_idx,
                    )
                    tb_logger.log_scalar(
                        "Metrics/Scene/PSNR_mean", scene_psnr, room_idx
                    )
                    tb_logger.log_scalar(
                        "Metrics/Scene/SSIM_mean", scene_ssim, room_idx
                    )
                    tb_logger.log_scalar(
                        "Metrics/Scene/NIQE_mean", scene_niqe, room_idx
                    )
                    tb_logger.log_scalar(
                        "Metrics/Scene/BRISQUE_mean", scene_brisque, room_idx
                    )
                    tb_logger.log_scalar("Metrics/Scene/FID", scene_fid, room_idx)

                    # Log depth metrics
                    if depth_count > 0:
                        scene_rmse = total_rmse / depth_count
                        scene_abs_rel = total_abs_rel / depth_count

                        tb_logger.log_scalar(
                            "Metrics/Scene/Depth_RMSE_mean", scene_rmse, room_idx
                        )
                        tb_logger.log_scalar(
                            "Metrics/Scene/Depth_AbsRel_mean", scene_abs_rel, room_idx
                        )

                        print(f"Scene average Depth RMSE: {scene_rmse:.4f}")
                        print(f"Scene average Depth Abs Rel: {scene_abs_rel:.4f}")

                    print(f"Scene average LPIPS: {scene_lpips}")
                    print(f"Scene average CLIP-Distance: {scene_clip_d}")
                    print(f"Scene average PSNR: {scene_psnr}")
                    print(f"Scene average SSIM: {scene_ssim}")
                    print(f"Scene average NIQE: {scene_niqe}")
                    print(f"Scene average BRISQUE: {scene_brisque}")
                    print(f"Scene FID: {scene_fid}")

                room_idx += 1

        tb_logger.close()

    else:
        raise Exception("Undefined dataset name!")


if __name__ == "__main__":
    main()
