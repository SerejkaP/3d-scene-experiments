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
from utils.metrics_computer import MetricsComputer


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
        metrics_computer = MetricsComputer(
            lpips_metric=lpips_metric,
            clip_metric=clip_metric,
            fid_metric=fid_metric,
            depth_metric=None,
            tb_logger=tb_logger,
        )

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

            # Compute and log panorama metrics
            pano_metrics = metrics_computer.compute_and_log_panorama_metrics(
                gt_path=current_pano_rgb_path,
                rendered_path=rendered_pano_path,
                step=i,
                scene_name=pano_name,
            )

            # Log panorama image comparison
            metrics_computer.log_image_comparison(
                tag=f"Images/Panorama/{pano_name}",
                gt_path=current_pano_rgb_path,
                rendered_path=rendered_pano_path,
                step=i,
                log_images=cfg.tensorboard.log_images,
            )

            camera_poses = [
                camera_json
                for camera_json in os.listdir(data_pose)
                if camera_json.startswith(f"{pano_name}_")
            ]

            camera_metrics_list = []
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

                # Compute camera metrics
                camera_metrics = metrics_computer.compute_camera_metrics(
                    gt_image_path=gt_image_path,
                    rendered_image_path=rendered_camera_path,
                )

                # Log camera metrics
                metrics_computer.log_camera_metrics(
                    metrics_dict=camera_metrics,
                    scene_name=pano_name,
                    camera_idx=camera_idx,
                )

                # Log camera image comparison
                metrics_computer.log_image_comparison(
                    tag=f"Images/Camera/{rendered_camera_subname}",
                    gt_path=gt_image_path,
                    rendered_path=rendered_camera_path,
                    step=i,
                    log_images=(
                        cfg.tensorboard.log_images
                        and camera_idx < cfg.tensorboard.max_images_per_scene
                    ),
                )

                # Collect metrics for scene aggregation
                camera_metrics_list.append(camera_metrics)
                gt_paths_for_fid.append(gt_image_path)
                render_paths_for_fid.append(rendered_camera_path)

            # Aggregate and log scene-level metrics
            scene_metrics = metrics_computer.aggregate_and_log_scene_metrics(
                metrics_list=camera_metrics_list,
                gt_paths=gt_paths_for_fid,
                rendered_paths=render_paths_for_fid,
                step=i,
            )

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
        metrics_computer = MetricsComputer(
            lpips_metric=lpips_metric,
            clip_metric=clip_metric,
            fid_metric=fid_metric,
            depth_metric=None,
            tb_logger=tb_logger,
        )
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
            metrics_computer.depth_metric = depth_metric
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

                    # Compute and log panorama metrics
                    pano_metrics = metrics_computer.compute_and_log_panorama_metrics(
                        gt_path=pano_rgb_path,
                        rendered_path=rendered_pano_path,
                        step=counter,
                        scene_name=room_name,
                    )

                    # Compute and log panorama depth metrics
                    if os.path.exists(gt_pano_depth_path) and os.path.exists(
                        rendered_pano_depth_path
                    ):
                        pano_depth_metrics = metrics_computer.compute_and_log_panorama_depth_metrics(
                            gt_depth_path=gt_pano_depth_path,
                            rendered_depth_path=rendered_pano_depth_path,
                            step=counter,
                            scene_name=room_name,
                            gt_depth_scale=1.0,
                        )

                    # Log panorama image comparison
                    metrics_computer.log_image_comparison(
                        tag=f"Images/Panorama/{room_name}",
                        gt_path=pano_rgb_path,
                        rendered_path=rendered_pano_path,
                        step=counter,
                        log_images=cfg.tensorboard.log_images,
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
        metrics_computer = MetricsComputer(
            lpips_metric=lpips_metric,
            clip_metric=clip_metric,
            fid_metric=fid_metric,
            depth_metric=depth_metric,
            tb_logger=tb_logger,
        )

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

                # Compute and log panorama metrics
                pano_metrics = metrics_computer.compute_and_log_panorama_metrics(
                    gt_path=pano_rgb_path,
                    rendered_path=rendered_pano_path,
                    step=room_idx,
                    scene_name=room_name,
                )

                # Compute and log panorama depth metrics
                if os.path.exists(gt_pano_depth_path) and os.path.exists(
                    rendered_pano_depth_path
                ):
                    pano_depth_metrics = metrics_computer.compute_and_log_panorama_depth_metrics(
                        gt_depth_path=gt_pano_depth_path,
                        rendered_depth_path=rendered_pano_depth_path,
                        step=room_idx,
                        scene_name=room_name,
                        gt_depth_scale=1.0 / 1000.0,
                    )

                # Log panorama image comparison
                metrics_computer.log_image_comparison(
                    tag=f"Images/Panorama/{room_name}",
                    gt_path=pano_rgb_path,
                    rendered_path=rendered_pano_path,
                    step=room_idx,
                    log_images=cfg.tensorboard.log_images,
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

                camera_metrics_list = []
                gt_paths_for_fid = []
                render_paths_for_fid = []

                # Depth metrics accumulators
                camera_depth_metrics_list = []

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

                    # Compute camera metrics
                    camera_metrics = metrics_computer.compute_camera_metrics(
                        gt_image_path=gt_image_path,
                        rendered_image_path=rendered_camera_path,
                    )

                    # Log camera metrics
                    metrics_computer.log_camera_metrics(
                        metrics_dict=camera_metrics,
                        scene_name=room_name,
                        camera_idx=camera_idx,
                    )

                    # Compute and log camera depth metrics
                    if os.path.exists(gt_depth_path) and os.path.exists(
                        rendered_depth_path
                    ):
                        camera_depth_metrics = metrics_computer.compute_camera_depth_metrics(
                            gt_depth_path=gt_depth_path,
                            rendered_depth_path=rendered_depth_path,
                            gt_depth_scale=1.0 / 1000.0,
                        )
                        if camera_depth_metrics:
                            metrics_computer.log_camera_depth_metrics(
                                depth_metrics=camera_depth_metrics,
                                scene_name=room_name,
                                camera_idx=camera_idx,
                            )
                            camera_depth_metrics_list.append(camera_depth_metrics)

                    # Log camera image comparison
                    metrics_computer.log_image_comparison(
                        tag=f"Images/Camera/{room_name}_{view_dir}",
                        gt_path=gt_image_path,
                        rendered_path=rendered_camera_path,
                        step=room_idx,
                        log_images=(
                            cfg.tensorboard.log_images
                            and camera_idx < cfg.tensorboard.max_images_per_scene
                        ),
                    )

                    # Collect metrics for scene aggregation
                    camera_metrics_list.append(camera_metrics)
                    gt_paths_for_fid.append(gt_image_path)
                    render_paths_for_fid.append(rendered_camera_path)

                # Aggregate and log scene-level metrics
                if len(camera_metrics_list) > 0:
                    scene_metrics = metrics_computer.aggregate_and_log_scene_metrics(
                        metrics_list=camera_metrics_list,
                        gt_paths=gt_paths_for_fid,
                        rendered_paths=render_paths_for_fid,
                        step=room_idx,
                    )

                    # Log scene depth metrics
                    if len(camera_depth_metrics_list) > 0:
                        scene_rmse = sum(m["rmse"] for m in camera_depth_metrics_list) / len(
                            camera_depth_metrics_list
                        )
                        scene_abs_rel = sum(
                            m["abs_rel"] for m in camera_depth_metrics_list
                        ) / len(camera_depth_metrics_list)

                        metrics_computer.tb_logger.log_scalar(
                            "Metrics/Scene/Depth_RMSE_mean", scene_rmse, room_idx
                        )
                        metrics_computer.tb_logger.log_scalar(
                            "Metrics/Scene/Depth_AbsRel_mean", scene_abs_rel, room_idx
                        )

                        print(f"Scene average Depth RMSE: {scene_rmse:.4f}")
                        print(f"Scene average Depth Abs Rel: {scene_abs_rel:.4f}")

                room_idx += 1

        tb_logger.close()

    else:
        raise Exception("Undefined dataset name!")


if __name__ == "__main__":
    main()
