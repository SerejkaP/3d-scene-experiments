import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import json
from tqdm import tqdm
from utils.gsplat_utils import render_camera, render_pano
from utils.worldgen_utils import worldgen_generate
from utils.dataset_2d3ds_utils import pose_json_by_image_path
from utils.metrics import (
    ClipDistanceMetric,
    compute_gt_metrics,
    LpipsMetric,
    compute_brisque,
    compute_niqe,
    FidMetric,
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

        # Close TensorBoard logger
        tb_logger.close()
    elif cfg.dataset.name == "ob3d":
        pass
    elif cfg.dataset.name == "structured3d":
        pass
    else:
        raise Exception("Undefined dataset name!")


if __name__ == "__main__":
    main()
