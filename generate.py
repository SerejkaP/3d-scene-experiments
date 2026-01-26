import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from worldgen import WorldGen
import numpy as np
import json
from PIL import Image
from gsplat_render import render
from utils.worldgen_utils import worldgen_generate
from utils.dataset_2d3ds_utils import pose_json_by_image_path


def create_gs(model_name, pano_path, save_path):
    if model_name == "worldgen":
        worldgen_generate(pano_path, save_path)
    else:
        raise Exception("Undefined model name!")


def render_2d3ds(ply_file, pano_json_path, camera_json_path, output_path):
    with open(pano_json_path, "r") as f:
        pano_data = json.load(f)

    with open(camera_json_path, "r") as f:
        camera_data = json.load(f)

    render(
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
        current_pano_rgb_name = pano_img_list[0]

        pano_json_path, pano_name = pose_json_by_image_path(
            current_pano_rgb_name, pano_pose
        )

        for i in range(cfg.generation_iters):
            print(f"Generate scene for {current_pano_rgb_name}")
            current_pano_rgb_path = os.path.join(pano_images, current_pano_rgb_name)
            save_path = os.path.join(cfg.save_path, f"{pano_name}_render.ply")
            create_gs(cfg.model.name, current_pano_rgb_path, save_path)

            camera_poses = [
                camera_json
                for camera_json in os.listdir(data_pose)
                if camera_json.startswith(f"{pano_name}_")
            ]
            for camera_pose in camera_poses:
                camera_json_path = os.path.join(data_pose, camera_pose)
                render_2d3ds(
                    save_path,
                    pano_json_path,
                    camera_json_path,
                    os.path.join(
                        cfg.save_path,
                        f"{'_'.join(camera_pose.split('_')[:7])}_render.png",
                    ),
                )

            current_pano_rgb_name = pano_img_list[i]
            pano_json_path, pano_name = pose_json_by_image_path(
                current_pano_rgb_name, pano_pose
            )
    if cfg.dataset.name == "ob3d":
        pass
    if cfg.dataset.name == "structured3d":
        pass


if __name__ == "__main__":
    main()
