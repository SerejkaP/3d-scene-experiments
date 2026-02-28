import os
import hydra
from omegaconf import DictConfig
from main import create_gs
from utils.tensorboard_logger import TensorBoardLogger
from utils.splits import load_split, filter_2d3ds_panos, filter_structured3d_rooms


def _limit_reached(counter, generation_iters):
    return generation_iters != -1 and counter >= generation_iters


def process_2d3ds(
    cfg,
    save_path,
    tb_logger: TensorBoardLogger,
    remove_generate: bool = False,
    restricted_data: list[str] = [],
):
    dataset_path = str(cfg.dataset.path)
    split_entries = load_split(str(cfg.splits_path), "2d3ds")
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
            generation_time = create_gs(cfg.model.name, pano_rgb_path, ply_path)
            tb_logger.log_scalar(
                "Performance/generation_time_seconds", generation_time, counter
            )
            print(f"Generation time: {generation_time}")
            generation_times.append(generation_time)
            if remove_generate and pano_rgb_path not in restricted_data:
                os.remove(ply_path)
            counter += 1

    return generation_times


def process_ob3d(
    cfg,
    save_path,
    tb_logger: TensorBoardLogger,
    remove_generate: bool = False,
    restricted_data: list[str] = [],
):
    avail_scenes = [
        "archiviz-flat",
        "barbershop",
        "classroom",
        "restroom",
        "san-miguel",
        "sun-temple",
    ]
    dataset_path = str(cfg.dataset.path)
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
                generation_time = create_gs(cfg.model.name, pano_rgb_path, ply_path)
                tb_logger.log_scalar(
                    "Performance/generation_time_seconds", generation_time, counter
                )
                print(f"Generation time: {generation_time}")
                generation_times.append(generation_time)
                if remove_generate and pano_rgb_path not in restricted_data:
                    os.remove(ply_path)
                counter += 1

    return generation_times


# def process_structured3d(cfg, save_path, tb_logger: TensorBoardLogger):
#     dataset_path = str(cfg.dataset.path)
#     split_entries = load_split(str(cfg.splits_path), "structured3d")
#     scenes = sorted([s for s in os.listdir(dataset_path) if s.startswith("scene_")])
#     generation_times = []
#     counter = 0
#     for scene in scenes:
#         if _limit_reached(counter, cfg.generation_iters):
#             break
#         rendering_path = os.path.join(dataset_path, scene, "2D_rendering")
#         if not os.path.isdir(rendering_path):
#             continue
#         rooms = sorted(os.listdir(rendering_path))
#         rooms = filter_structured3d_rooms(rooms, scene, split_entries)
#         for room in rooms:
#             if _limit_reached(counter, cfg.generation_iters):
#                 break
#             panorama_dir = os.path.join(rendering_path, room, "panorama")
#             pano_rgb_path = os.path.join(panorama_dir, "full", "rgb_rawlight.png")
#             if not os.path.exists(pano_rgb_path):
#                 print(f"Skipping {scene}/{room}: missing panorama")
#                 continue

#             room_save_path = os.path.join(save_path, scene, room)
#             os.makedirs(room_save_path, exist_ok=True)
#             ply_path = os.path.join(room_save_path, "scene.ply")

#             room_name = f"{scene}_{room}"
#             print(f"[structured3d] Generate scene for {room_name}")
#             generation_time = create_gs(cfg.model.name, pano_rgb_path, ply_path)
#             tb_logger.log_scalar(
#                 "Performance/generation_time_seconds", generation_time, counter
#             )
#             print(f"Generation time: {generation_time}")
#             generation_times.append(generation_time)
#             counter += 1

#     return generation_times


@hydra.main(config_path="conf", config_name="config", version_base=None)
def generate(cfg: DictConfig):
    tb_logger = TensorBoardLogger(
        log_dir=cfg.tensorboard.log_dir,
        enabled=cfg.tensorboard.enabled,
        flush_secs=cfg.tensorboard.flush_secs,
        max_image_size=2048,
    )

    dataset_name = str(cfg.dataset.name)
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}\n")

    save_path = os.path.join(cfg.save_path, cfg.model.name, dataset_name)
    os.makedirs(save_path, exist_ok=True)

    if dataset_name == "2d3ds":
        dataset_times = process_2d3ds(cfg, save_path, tb_logger)
    elif dataset_name == "ob3ds":
        dataset_times = process_ob3d(cfg, save_path, tb_logger)
    else:
        raise Exception(f"Unknown dataset: {dataset_name}!")
    if dataset_times:
        avg = sum(dataset_times) / len(dataset_times)
        print(
            f"\n[{dataset_name}] Average generation time: {avg:.2f}s ({len(dataset_times)} scenes)"
        )
        tb_logger.log_scalar(f"Performance/{dataset_name}/avg_generation_time", avg, 0)

    tb_logger.close()


if __name__ == "__main__":
    generate()
