import os
import sys
import shutil
import tempfile

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "HunyuanWorld-Mirror"),
)

import hydra
from omegaconf import DictConfig
from generate_scene import generate_scene
from utils.tensorboard_logger import TensorBoardLogger
from utils.splits import load_split, filter_2d3ds_panos, filter_structured3d_rooms


def _limit_reached(counter, generation_iters):
    return generation_iters != -1 and counter >= generation_iters


def _create_hunyuanworld_mirror(pano_path, ply_path, target_size, fps, pretrained):
    """Generate a HunyuanWorld-Mirror scene from a single panorama, move result to ply_path."""
    parent_dir = os.path.dirname(ply_path)
    # HunyuanWorld-Mirror expects a directory of images, so put the pano in a temp dir.
    img_dir = tempfile.mkdtemp(dir=parent_dir)
    out_dir = tempfile.mkdtemp(dir=parent_dir)
    try:
        shutil.copy2(pano_path, os.path.join(img_dir, os.path.basename(pano_path)))
        src_ply, elapsed = generate_scene(
            img_dir, out_dir, target_size=target_size, fps=fps, pretrained=pretrained
        )
        shutil.move(src_ply, ply_path)
    finally:
        shutil.rmtree(img_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
    return elapsed


def process_2d3ds(cfg, save_path, tb_logger: TensorBoardLogger):
    dataset_path = str(cfg.dataset.path)
    split_entries = load_split(str(cfg.splits_path), "2d3ds")
    target_size = cfg.model.params.target_size
    fps = cfg.model.params.fps
    pretrained = cfg.model.params.pretrained
    areas = sorted([s for s in os.listdir(dataset_path) if s.startswith("area_")])
    generation_times = []
    counter = 0
    for area in areas:
        if _limit_reached(counter, cfg.generation_iters):
            break
        pano_images = os.path.join(dataset_path, area, "pano", "rgb")
        pano_img_list = sorted(
            [img for img in os.listdir(pano_images) if img.endswith(".png")]
        )
        pano_img_list = filter_2d3ds_panos(pano_img_list, area, split_entries)
        if len(pano_img_list) == 0:
            continue

        for pano_name in pano_img_list:
            if _limit_reached(counter, cfg.generation_iters):
                break
            pano_rgb_path = os.path.join(pano_images, pano_name)
            scene_name = os.path.splitext(pano_name)[0]
            area_save_path = os.path.join(save_path, area)
            os.makedirs(area_save_path, exist_ok=True)
            ply_path = os.path.join(area_save_path, f"{scene_name}_render.ply")

            print(f"[2d3ds] Generate scene for {pano_name}")
            generation_time = _create_hunyuanworld_mirror(
                pano_rgb_path, ply_path, target_size, fps, pretrained
            )
            tb_logger.log_scalar(
                "Performance/generation_time_seconds", generation_time, counter
            )
            print(f"Generation time: {generation_time}")
            generation_times.append(generation_time)
            counter += 1

    return generation_times


def process_ob3d(cfg, save_path, tb_logger: TensorBoardLogger):
    avail_scenes = [
        "archiviz-flat",
        "barbershop",
        "classroom",
        "restroom",
        "san-miguel",
        "sun-temple",
    ]
    dataset_path = str(cfg.dataset.path)
    target_size = cfg.model.params.target_size
    fps = cfg.model.params.fps
    pretrained = cfg.model.params.pretrained
    generation_times = []
    counter = 0
    for scene in avail_scenes:
        if _limit_reached(counter, cfg.generation_iters):
            break
        for scene_type in [
            # "Egocentric",
            "Non-Egocentric"
        ]:
            if _limit_reached(counter, cfg.generation_iters):
                break
            scene_path = os.path.join(dataset_path, scene, scene_type)
            images_path = os.path.join(scene_path, "images")
            test_file = os.path.join(scene_path, "test.txt")
            if not os.path.exists(test_file):
                continue
            with open(test_file, "r") as f:
                test_data = [int(line.strip()) for line in f.readlines()]

            save_scene_path = os.path.join(save_path, scene, scene_type)
            os.makedirs(save_scene_path, exist_ok=True)

            for t in test_data:
                if _limit_reached(counter, cfg.generation_iters):
                    break
                t_scene = f"{str(t).zfill(5)}"
                pano_rgb_path = os.path.join(images_path, f"{t_scene}_rgb.png")
                ply_path = os.path.join(save_scene_path, f"{t_scene}_render.ply")

                print(f"[ob3d] Generate scene for {scene}/{scene_type}/{t_scene}")
                generation_time = _create_hunyuanworld_mirror(
                    pano_rgb_path, ply_path, target_size, fps, pretrained
                )
                tb_logger.log_scalar(
                    "Performance/generation_time_seconds", generation_time, counter
                )
                print(f"Generation time: {generation_time}")
                generation_times.append(generation_time)
                counter += 1

    return generation_times


def process_structured3d(cfg, save_path, tb_logger: TensorBoardLogger):
    dataset_path = str(cfg.dataset.path)
    split_entries = load_split(str(cfg.splits_path), "structured3d")
    target_size = cfg.model.params.target_size
    fps = cfg.model.params.fps
    pretrained = cfg.model.params.pretrained
    scenes = sorted([s for s in os.listdir(dataset_path) if s.startswith("scene_")])
    generation_times = []
    counter = 0
    for scene in scenes:
        if _limit_reached(counter, cfg.generation_iters):
            break
        rendering_path = os.path.join(dataset_path, scene, "2D_rendering")
        if not os.path.isdir(rendering_path):
            continue
        rooms = sorted(os.listdir(rendering_path))
        rooms = filter_structured3d_rooms(rooms, scene, split_entries)
        for room in rooms:
            if _limit_reached(counter, cfg.generation_iters):
                break
            panorama_dir = os.path.join(rendering_path, room, "panorama")
            pano_rgb_path = os.path.join(panorama_dir, "full", "rgb_rawlight.png")
            if not os.path.exists(pano_rgb_path):
                print(f"Skipping {scene}/{room}: missing panorama")
                continue

            room_save_path = os.path.join(save_path, scene, room)
            os.makedirs(room_save_path, exist_ok=True)
            ply_path = os.path.join(room_save_path, "scene.ply")

            room_name = f"{scene}_{room}"
            print(f"[structured3d] Generate scene for {room_name}")
            generation_time = _create_hunyuanworld_mirror(
                pano_rgb_path, ply_path, target_size, fps, pretrained
            )
            tb_logger.log_scalar(
                "Performance/generation_time_seconds", generation_time, counter
            )
            print(f"Generation time: {generation_time}")
            generation_times.append(generation_time)
            counter += 1

    return generation_times


DATASET_PROCESSORS = {
    "2d3ds": process_2d3ds,
    "ob3d": process_ob3d,
    "structured3d": process_structured3d,
}


@hydra.main(config_path="conf", config_name="config", version_base=None)
def generate(cfg: DictConfig):
    tb_logger = TensorBoardLogger(
        log_dir=cfg.tensorboard.log_dir,
        enabled=cfg.tensorboard.enabled,
        flush_secs=cfg.tensorboard.flush_secs,
        max_image_size=2048,
    )

    dataset_name = cfg.dataset.name
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}\n")

    save_path = os.path.join(cfg.save_path, cfg.model.name, dataset_name)
    os.makedirs(save_path, exist_ok=True)

    processor = DATASET_PROCESSORS.get(dataset_name)
    if processor:
        dataset_times = processor(cfg, save_path, tb_logger)
        if dataset_times:
            avg = sum(dataset_times) / len(dataset_times)
            print(
                f"\n[{dataset_name}] Average generation time: {avg:.2f}s ({len(dataset_times)} scenes)"
            )
            tb_logger.log_scalar(
                f"Performance/{dataset_name}/avg_generation_time", avg, 0
            )
    else:
        print(f"Unknown dataset: {dataset_name}, skipping")

    tb_logger.close()


if __name__ == "__main__":
    generate()
