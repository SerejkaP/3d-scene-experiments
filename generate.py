import os
import hydra
from omegaconf import DictConfig
import numpy as np
import json
from gsplat_render import render_camera
from utils.worldgen_utils import worldgen_generate
from utils.dataset_2d3ds_utils import pose_json_by_image_path
from utils.metrics import compute_gt_metrics


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

        num_iters = min(cfg.generation_iters, len(pano_img_list))
        for i in range(num_iters):
            print(f"Generate scene for {current_pano_rgb_name}")
            current_pano_rgb_path = os.path.join(pano_images, current_pano_rgb_name)
            ply_render_path = os.path.join(cfg.save_path, f"{pano_name}_render.ply")
            # Generation time metric
            generation_time = create_gs(
                cfg.model.name, current_pano_rgb_path, ply_render_path
            )

            camera_poses = [
                camera_json
                for camera_json in os.listdir(data_pose)
                if camera_json.startswith(f"{pano_name}_")
            ]

            total_psnr = 0
            total_ssim = 0
            for camera_pose in camera_poses:
                camera_json_path = os.path.join(data_pose, camera_pose)
                rendered_camera_subname = "_".join(camera_pose.split("_")[:-1])
                rendered_camera = f"{rendered_camera_subname}_render.png"
                rendered_camera_path = os.path.join(cfg.save_path, rendered_camera)
                render_2d3ds(
                    ply_render_path,
                    pano_json_path,
                    camera_json_path,
                    rendered_camera_path,
                )

                gt_image_path = os.path.join(
                    data_images, f"{rendered_camera_subname}_rgb.png"
                )
                print(gt_image_path, rendered_camera_path)
                psnr, ssim = compute_gt_metrics(gt_image_path, rendered_camera_path)
                total_psnr += psnr
                total_ssim += ssim
            scene_psnr = total_psnr / len(camera_poses)
            scene_ssim = total_ssim / len(camera_poses)
            print("PSNR: ", scene_psnr)
            print("SSIM: ", scene_ssim)

            current_pano_rgb_name = pano_img_list[i]
            pano_json_path, pano_name = pose_json_by_image_path(
                current_pano_rgb_name, pano_pose
            )
    if cfg.dataset.name == "ob3d":
        pass
    if cfg.dataset.name == "structured3d":
        pass
    else:
        raise Exception("Undefined dataset name!")


if __name__ == "__main__":
    main()
