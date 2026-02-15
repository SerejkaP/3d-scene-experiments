import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm
from main import render_2d3ds, render_structured3d
from utils.gsplat_utils import render_pano
from utils.dataset_2d3ds_utils import pose_json_by_image_path
from utils.metrics import ClipDistanceMetric, LpipsMetric, FidMetric, DepthMetrics
from utils.tensorboard_logger import TensorBoardLogger
from utils.metrics_computer import MetricsComputer
from utils.splits import load_split, filter_2d3ds_panos, filter_structured3d_rooms


def evaluate_2d3ds(cfg, save_path, tb_logger: TensorBoardLogger):
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
    dataset_path = str(cfg.dataset.path)
    split_entries = load_split(str(cfg.splits_path), "2d3ds")
    areas = sorted([s for s in os.listdir(dataset_path) if s.startswith("area_")])
    counter = 0
    for area in areas:
        if counter >= cfg.generation_iters:
            break
        pano_images = os.path.join(dataset_path, area, "pano", "rgb")
        pano_pose = os.path.join(dataset_path, area, "pano", "pose")
        data_pose = os.path.join(dataset_path, area, "data", "pose")
        data_images = os.path.join(dataset_path, area, "data", "rgb")
        pano_img_list = sorted([img for img in os.listdir(pano_images) if img.endswith(".png")])
        pano_img_list = filter_2d3ds_panos(pano_img_list, area, split_entries)
        if len(pano_img_list) == 0:
            continue

        for current_pano_rgb_name in pano_img_list:
            if counter >= cfg.generation_iters:
                break

            scene_name = os.path.splitext(current_pano_rgb_name)[0]
            current_pano_rgb_path = os.path.join(pano_images, current_pano_rgb_name)
            area_save_path = os.path.join(save_path, area)
            ply_render_path = os.path.join(area_save_path, f"{scene_name}_render.ply")

            if not os.path.exists(ply_render_path):
                print(f"[2d3ds] PLY not found, skipping: {ply_render_path}")
                continue

            pano_json_path, pano_name = pose_json_by_image_path(
                current_pano_rgb_name, pano_pose
            )
            rendered_pano_path = os.path.join(area_save_path, f"{scene_name}_pano.png")

            print(f"[2d3ds] Evaluate scene for {current_pano_rgb_name}")

            render_pano(
                ply_render_path,
                [0, 0, 0],
                cfg.dataset.pano_width,
                rendered_pano_path,
            )

            metrics_computer.compute_and_log_panorama_metrics(
                gt_path=current_pano_rgb_path,
                rendered_path=rendered_pano_path,
                step=counter,
                scene_name=pano_name,
            )

            metrics_computer.log_image_comparison(
                tag=f"Images/Panorama/{pano_name}",
                gt_path=current_pano_rgb_path,
                rendered_path=rendered_pano_path,
                step=counter,
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
                rendered_camera_path = os.path.join(area_save_path, rendered_camera)

                render_2d3ds(
                    ply_render_path,
                    pano_json_path,
                    camera_json_path,
                    rendered_camera_path,
                )

                gt_image_path = os.path.join(
                    data_images, f"{rendered_camera_subname}_rgb.png"
                )

                camera_metrics = metrics_computer.compute_camera_metrics(
                    gt_image_path=gt_image_path,
                    rendered_image_path=rendered_camera_path,
                )

                metrics_computer.log_camera_metrics(
                    metrics_dict=camera_metrics,
                    scene_name=pano_name,
                    camera_idx=camera_idx,
                )

                metrics_computer.log_image_comparison(
                    tag=f"Images/Camera/{rendered_camera_subname}",
                    gt_path=gt_image_path,
                    rendered_path=rendered_camera_path,
                    step=counter,
                    log_images=(
                        cfg.tensorboard.log_images
                        and camera_idx < cfg.tensorboard.max_images_per_scene
                    ),
                )

                camera_metrics_list.append(camera_metrics)
                gt_paths_for_fid.append(gt_image_path)
                render_paths_for_fid.append(rendered_camera_path)

            metrics_computer.aggregate_and_log_scene_metrics(
                metrics_list=camera_metrics_list,
                gt_paths=gt_paths_for_fid,
                rendered_paths=render_paths_for_fid,
                step=counter,
            )

            counter += 1


def evaluate_ob3d(cfg, save_path, tb_logger: TensorBoardLogger):
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
    dataset_path = str(cfg.dataset.path)
    counter = 0
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
            scene_path = os.path.join(dataset_path, scene, scene_type)
            images_path = os.path.join(scene_path, "images")
            depth_path = os.path.join(scene_path, "depths")
            test_file = os.path.join(scene_path, "test.txt")
            if not os.path.exists(test_file):
                continue
            with open(test_file, "r") as f:
                test_data = [int(line.strip()) for line in f.readlines()]

            save_scene_path = os.path.join(save_path, scene, scene_type)

            for t in test_data:
                if counter >= cfg.generation_iters:
                    break
                t_scene = f"{str(t).zfill(5)}"
                ply_render_path = os.path.join(save_scene_path, f"{t_scene}_render.ply")

                if not os.path.exists(ply_render_path):
                    print(f"[ob3d] PLY not found, skipping: {ply_render_path}")
                    continue

                rendered_pano_path = os.path.join(
                    save_scene_path, f"{t_scene}_pano.png"
                )
                rendered_pano_depth_path = os.path.join(
                    save_scene_path, f"{t_scene}_depth.exr"
                )
                pano_rgb_path = os.path.join(images_path, f"{t_scene}_rgb.png")
                gt_pano_depth_path = os.path.join(depth_path, f"{t_scene}_depth.exr")
                room_name = f"{scene}/{scene_type}/{t_scene}"

                print(f"[ob3d] Evaluate scene for {room_name}")

                render_pano(
                    ply_render_path,
                    [0, 0, 0],
                    cfg.dataset.pano_width,
                    rendered_pano_path,
                    rendered_pano_depth_path,
                )

                metrics_computer.compute_and_log_panorama_metrics(
                    gt_path=pano_rgb_path,
                    rendered_path=rendered_pano_path,
                    step=counter,
                    scene_name=room_name,
                )

                if os.path.exists(gt_pano_depth_path) and os.path.exists(
                    rendered_pano_depth_path
                ):
                    metrics_computer.compute_and_log_panorama_depth_metrics(
                        gt_depth_path=gt_pano_depth_path,
                        rendered_depth_path=rendered_pano_depth_path,
                        step=counter,
                        scene_name=room_name,
                        gt_depth_scale=1.0,
                    )

                metrics_computer.log_image_comparison(
                    tag=f"Images/Panorama/{room_name}",
                    gt_path=pano_rgb_path,
                    rendered_path=rendered_pano_path,
                    step=counter,
                    log_images=cfg.tensorboard.log_images,
                )

                counter += 1


def evaluate_structured3d(cfg, save_path, tb_logger: TensorBoardLogger):
    dataset_path = str(cfg.dataset.path)
    split_entries = load_split(str(cfg.splits_path), "structured3d")
    scenes = sorted([s for s in os.listdir(dataset_path) if s.startswith("scene_")])

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
        rooms = filter_structured3d_rooms(rooms, scene, split_entries)
        for room in rooms:
            if room_idx >= cfg.generation_iters:
                break

            room_path = os.path.join(rendering_path, room)
            panorama_dir = os.path.join(room_path, "panorama")
            perspective_dir = os.path.join(room_path, "perspective", "full")

            camera_xyz_file = os.path.join(panorama_dir, "camera_xyz.txt")
            pano_rgb_path = os.path.join(panorama_dir, "full", "rgb_rawlight.png")

            if not os.path.exists(camera_xyz_file) or not os.path.exists(pano_rgb_path):
                print(f"Skipping {scene}/{room}: missing data")
                continue

            room_save_path = os.path.join(save_path, scene, room)
            ply_render_path = os.path.join(room_save_path, "scene.ply")

            if not os.path.exists(ply_render_path):
                print(f"[structured3d] PLY not found, skipping: {ply_render_path}")
                continue

            with open(camera_xyz_file, "r") as f:
                pano_position = np.array([float(x) for x in f.read().strip().split()])

            room_name = f"{scene}_{room}"
            rendered_pano_path = os.path.join(room_save_path, "pano_render.png")
            rendered_pano_depth_path = os.path.join(room_save_path, "pano_depth.exr")
            gt_pano_depth_path = os.path.join(panorama_dir, "full", "depth.png")

            print(f"[structured3d] Evaluate scene for {room_name}")

            render_pano(
                ply_render_path,
                [0, 0, 0],
                cfg.dataset.pano_width,
                rendered_pano_path,
                output_depth_path=rendered_pano_depth_path,
            )

            metrics_computer.compute_and_log_panorama_metrics(
                gt_path=pano_rgb_path,
                rendered_path=rendered_pano_path,
                step=room_idx,
                scene_name=room_name,
            )

            if os.path.exists(gt_pano_depth_path) and os.path.exists(
                rendered_pano_depth_path
            ):
                metrics_computer.compute_and_log_panorama_depth_metrics(
                    gt_depth_path=gt_pano_depth_path,
                    rendered_depth_path=rendered_pano_depth_path,
                    step=room_idx,
                    scene_name=room_name,
                    gt_depth_scale=1.0 / 1000.0,
                )

            metrics_computer.log_image_comparison(
                tag=f"Images/Panorama/{room_name}",
                gt_path=pano_rgb_path,
                rendered_path=rendered_pano_path,
                step=room_idx,
                log_images=cfg.tensorboard.log_images,
            )

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

                camera_metrics = metrics_computer.compute_camera_metrics(
                    gt_image_path=gt_image_path,
                    rendered_image_path=rendered_camera_path,
                )

                metrics_computer.log_camera_metrics(
                    metrics_dict=camera_metrics,
                    scene_name=room_name,
                    camera_idx=camera_idx,
                )

                if os.path.exists(gt_depth_path) and os.path.exists(
                    rendered_depth_path
                ):
                    camera_depth_metrics = (
                        metrics_computer.compute_camera_depth_metrics(
                            gt_depth_path=gt_depth_path,
                            rendered_depth_path=rendered_depth_path,
                            gt_depth_scale=1.0 / 1000.0,
                        )
                    )
                    if camera_depth_metrics:
                        metrics_computer.log_camera_depth_metrics(
                            depth_metrics=camera_depth_metrics,
                            scene_name=room_name,
                            camera_idx=camera_idx,
                        )
                        camera_depth_metrics_list.append(camera_depth_metrics)

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

                camera_metrics_list.append(camera_metrics)
                gt_paths_for_fid.append(gt_image_path)
                render_paths_for_fid.append(rendered_camera_path)

            if len(camera_metrics_list) > 0:
                metrics_computer.aggregate_and_log_scene_metrics(
                    metrics_list=camera_metrics_list,
                    gt_paths=gt_paths_for_fid,
                    rendered_paths=render_paths_for_fid,
                    step=room_idx,
                )

                if len(camera_depth_metrics_list) > 0:
                    scene_rmse = sum(
                        m["rmse"] for m in camera_depth_metrics_list
                    ) / len(camera_depth_metrics_list)
                    scene_abs_rel = sum(
                        m["abs_rel"] for m in camera_depth_metrics_list
                    ) / len(camera_depth_metrics_list)

                    tb_logger.log_scalar(
                        "Metrics/Scene/Depth_RMSE_mean", scene_rmse, room_idx
                    )
                    tb_logger.log_scalar(
                        "Metrics/Scene/Depth_AbsRel_mean", scene_abs_rel, room_idx
                    )

                    print(f"Scene average Depth RMSE: {scene_rmse:.4f}")
                    print(f"Scene average Depth Abs Rel: {scene_abs_rel:.4f}")

            room_idx += 1


DATASET_EVALUATORS = {
    "2d3ds": evaluate_2d3ds,
    "ob3d": evaluate_ob3d,
    "structured3d": evaluate_structured3d,
}


@hydra.main(config_path="conf", config_name="config", version_base=None)
def evaluate(cfg: DictConfig):
    conf_dir = os.path.join(hydra.utils.get_original_cwd(), "conf", "dataset")
    dataset_files = sorted(
        [f[:-5] for f in os.listdir(conf_dir) if f.endswith(".yaml")]
    )

    tb_logger = TensorBoardLogger(
        log_dir=cfg.tensorboard.log_dir,
        enabled=cfg.tensorboard.enabled,
        flush_secs=cfg.tensorboard.flush_secs,
        max_image_size=2048,
    )

    for dataset_name in dataset_files:
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*60}\n")

        dataset_cfg = OmegaConf.load(os.path.join(conf_dir, f"{dataset_name}.yaml"))
        run_cfg = OmegaConf.merge(cfg, {"dataset": dataset_cfg})

        save_path = os.path.join(cfg.save_path, cfg.model.name, dataset_name)

        if not os.path.isdir(save_path):
            print(f"Output directory not found, skipping: {save_path}")
            continue

        evaluator = DATASET_EVALUATORS.get(dataset_name)
        if evaluator:
            evaluator(run_cfg, save_path, tb_logger)
        else:
            print(f"Unknown dataset: {dataset_name}, skipping")

    tb_logger.close()


if __name__ == "__main__":
    evaluate()
