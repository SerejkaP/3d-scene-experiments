"""
Экспорт панорам тестовой выборки с сохранением оригинальной структуры папок.

Читает test.txt для каждого датасета и копирует (или создаёт symlink)
панорамные изображения в выходную директорию.

Результат:
  test/
  ├── 2d3ds/
  │   ├── area_1/
  │   │   └── pano/rgb/
  │   │       ├── camera_..._rgb.png
  │   │       └── ...
  │   └── area_2/
  │       └── ...
  ├── structured3d/
  │   ├── scene_00001/
  │   │   └── 2D_rendering/
  │   │       └── 906325/
  │   │           └── panorama/full/
  │   │               └── rgb_rawlight.png
  │   └── ...
  └── ob3d/
      ├── archiviz-flat/
      │   ├── Egocentric/
      │   │   └── images/
      │   │       ├── 00003_rgb.png
      │   │       └── ...
      │   └── Non-Egocentric/
      │       └── images/
      │           └── ...
      └── barbershop/
          └── ...
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np
import OpenEXR
import Imath
import py360convert
from tqdm import tqdm


# Грани кубмапы: (имя, yaw°, pitch°) — совпадает с соглашением gsplat_utils/_get_cubemap_views
_CUBEMAP_FACES = [
    ("front", 0.0, 0.0),
    ("right", 90.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", 270.0, 0.0),
    ("top", 0.0, 90.0),
    ("bottom", 0.0, -90.0),
]


def _yaw_pitch_to_R(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """World-to-camera rotation (та же конвенция, что в pano_virtual_cameras.py)."""
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    forward = np.array(
        [
            -np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            -np.cos(yaw) * np.cos(pitch),
        ]
    )
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = (
            np.array([0.0, 0.0, 1.0]) if forward[1] > 0 else np.array([0.0, 0.0, -1.0])
        )
    z = -forward
    x = np.cross(world_up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)


def _load_exr_depth(path: str) -> np.ndarray:
    f = OpenEXR.InputFile(path)
    dw = f.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    for ch in ["Z", "V", "R", "G", "B"]:
        if ch in f.header()["channels"]:
            return np.frombuffer(f.channel(ch, pt), dtype=np.float32).reshape(h, w)
    raise ValueError(f"Нет подходящего канала в {path}")


def _save_exr_depth(arr: np.ndarray, path: str) -> None:
    h, w = arr.shape
    header = OpenEXR.Header(w, h)
    header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    f = OpenEXR.OutputFile(path, header)
    f.writePixels({"Z": arr.astype(np.float32).tobytes()})
    f.close()


def _read_ob3d_world_pos(cam_json_path: str) -> list:
    """Reads the world position (camera center) from an OB3D camera JSON.

    OB3D stores w2c extrinsics: rotation R_w2c and translation t_w2c.
    The camera center in world coordinates is: pos = -R_w2c.T @ t_w2c
    """
    with open(cam_json_path) as f:
        data = json.load(f)[0]
    R_w2c = np.array(data["extrinsics"]["rotation"])
    t_w2c = np.array(data["extrinsics"]["translation"])
    return (-R_w2c.T @ t_w2c).tolist()


def _export_cubemap_data(
    pano_rgb_path: str,
    depth_path: str | None,
    out_dir: str,
    prefix: str,
    face_size: int = 512,
    pano_world_pos: list | None = None,
) -> None:
    """
    Нарезает equirectangular панораму на 6 граней кубмапы, сохраняет
    соответствующие depth-карты и JSON с параметрами камеры.

    Сохраняет в out_dir:
      {prefix}_{face}.png       — RGB грань
      {prefix}_{face}_depth.exr — depth грань
      {prefix}_{face}.json      — K, R, width, height для render_virtual_camera
    """
    pano_bgr = cv2.imread(pano_rgb_path)
    if pano_bgr is None:
        print(f"  [warn] не удалось загрузить: {pano_rgb_path}")
        return
    pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

    depth = None
    if depth_path and os.path.exists(depth_path):
        depth = _load_exr_depth(depth_path)

    os.makedirs(out_dir, exist_ok=True)

    fov = 90.0  # кубмапа — 90° FOV на грань
    for face_name, yaw, pitch in _CUBEMAP_FACES:
        # RGB
        face_rgb = py360convert.e2p(
            pano_rgb,
            fov_deg=(fov, fov),
            u_deg=yaw,
            v_deg=pitch,
            out_hw=(face_size, face_size),
            in_rot_deg=0,
            mode="bilinear",
        )
        cv2.imwrite(
            os.path.join(out_dir, f"{prefix}_{face_name}.png"),
            cv2.cvtColor(face_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR),
        )

        # Depth
        if depth is not None:
            depth_3ch = np.stack([depth, depth, depth], axis=-1)
            face_depth = py360convert.e2p(
                depth_3ch,
                fov_deg=(fov, fov),
                u_deg=yaw,
                v_deg=pitch,
                out_hw=(face_size, face_size),
                in_rot_deg=0,
                mode="bilinear",
            )[:, :, 0]
            _save_exr_depth(
                face_depth, os.path.join(out_dir, f"{prefix}_{face_name}_depth.exr")
            )

        # Camera JSON
        R = _yaw_pitch_to_R(yaw, pitch)
        focal = face_size / (2.0 * np.tan(np.radians(fov) / 2.0))
        cx = cy = face_size / 2.0
        cam_info = {
            "face": face_name,
            "fov_h": fov,
            "fov_v": fov,
            "width": face_size,
            "height": face_size,
            "K": [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            "R": R.tolist(),
            "pano_world_pos": (
                pano_world_pos if pano_world_pos is not None else [0.0, 0.0, 0.0]
            ),
        }
        with open(os.path.join(out_dir, f"{prefix}_{face_name}.json"), "w") as fp:
            json.dump(cam_info, fp, indent=2)


def load_test_entries(splits_path: str, dataset_name: str) -> list[str]:
    test_file = os.path.join(splits_path, dataset_name, "test.txt")
    if not os.path.exists(test_file):
        print(f"  [warn] {test_file} не найден, пропуск")
        return []
    with open(test_file, "r") as f:
        entries = [line.strip() for line in f if line.strip()]
    print(f"  Загружено {len(entries)} записей из {test_file}")
    return entries


def _copy_or_link(src: str, dst: str, symlink: bool) -> None:
    if not os.path.exists(src):
        print(f"  [warn] не найден: {src}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if symlink:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def export_2d3ds(dataset_path: str, entries: list[str], output_dir: str, symlink: bool):
    """Копирует данные 2D-3D-Semantics, необходимые для evaluate.py.

    Для каждой записи экспортирует:
      pano/rgb/   — RGB панорама
      pano/depth/ — depth карта панорамы
      pano/pose/  — pose JSON панорамы
      data/pose/  — pose JSON perspective-камер данной панорамы
      data/rgb/   — RGB perspective-виды
      data/depth/ — depth карты perspective-видов

    Формат записи: area_N/filename.png
    """
    for entry in tqdm(entries):
        area, filename = entry.split("/", 1)

        # pano_name: первые 5 частей имени файла (camera_HASH_room_N_frame)
        pano_name = "_".join(filename.split("_")[:5])

        # pano/rgb
        _copy_or_link(
            os.path.join(dataset_path, area, "pano", "rgb", filename),
            os.path.join(output_dir, area, "pano", "rgb", filename),
            symlink,
        )

        # pano/depth
        depth_filename = filename.replace("_rgb.png", "_depth.png")
        _copy_or_link(
            os.path.join(dataset_path, area, "pano", "depth", depth_filename),
            os.path.join(output_dir, area, "pano", "depth", depth_filename),
            symlink,
        )

        # pano/pose
        pose_filename = f"{pano_name}_equirectangular_domain_pose.json"
        _copy_or_link(
            os.path.join(dataset_path, area, "pano", "pose", pose_filename),
            os.path.join(output_dir, area, "pano", "pose", pose_filename),
            symlink,
        )

        # data/pose, data/rgb, data/depth
        data_pose_src = os.path.join(dataset_path, area, "data", "pose")
        if not os.path.isdir(data_pose_src):
            continue

        for camera_pose in tqdm(os.listdir(data_pose_src)):
            # Only data used for our panos
            if not camera_pose.startswith(f"{pano_name}_"):
                continue

            # data/pose
            _copy_or_link(
                os.path.join(data_pose_src, camera_pose),
                os.path.join(output_dir, area, "data", "pose", camera_pose),
                symlink,
            )

            # subname = имя без последнего компонента "_pose.json"
            subname = "_".join(camera_pose.split("_")[:-1])

            # data/rgb
            _copy_or_link(
                os.path.join(dataset_path, area, "data", "rgb", f"{subname}_rgb.png"),
                os.path.join(output_dir, area, "data", "rgb", f"{subname}_rgb.png"),
                symlink,
            )

            # data/depth
            _copy_or_link(
                os.path.join(
                    dataset_path, area, "data", "depth", f"{subname}_depth.png"
                ),
                os.path.join(output_dir, area, "data", "depth", f"{subname}_depth.png"),
                symlink,
            )

    print(f"  Экспортировано {len(entries)} записей -> {output_dir}")


def export_structured3d(
    dataset_path: str, entries: list[str], output_dir: str, symlink: bool
):
    """Копирует панорамы Structured3D.

    Формат записи: scene_XXXXX/room_id
    Источник:      {dataset}/scene_XXXXX/2D_rendering/room_id/panorama/full/rgb_rawlight.png
    Назначение:    {output}/scene_XXXXX/2D_rendering/room_id/panorama/full/rgb_rawlight.png
    """
    for entry in entries:
        scene, room = entry.split("/", 1)
        src = os.path.join(
            dataset_path,
            scene,
            "2D_rendering",
            room,
            "panorama",
            "full",
            "rgb_rawlight.png",
        )
        dst = os.path.join(
            output_dir,
            scene,
            "2D_rendering",
            room,
            "panorama",
            "full",
            "rgb_rawlight.png",
        )

        if not os.path.exists(src):
            print(f"  [warn] не найден: {src}")
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if symlink:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)

    print(f"  Экспортировано {len(entries)} панорам -> {output_dir}")


AVAIL_OB3D_SCENES = [
    "archiviz-flat",
    "barbershop",
    "classroom",
    "restroom",
    "san-miguel",
    "sun-temple",
]


def export_ob3d(dataset_path: str, output_dir: str, symlink: bool):
    """Копирует данные OB3D из тестовых сплитов, хранящихся внутри датасета.

    Источник test.txt: {dataset}/scene/scene_type/test.txt
    Содержимое:        целочисленные индексы кадров, по одному на строку
    Для каждого индекса экспортирует:
      images/{index:05d}_rgb.png
      depths/{index:05d}_depth.exr
      cameras/{index:05d}_cam.json
    """
    total = 0
    for scene in AVAIL_OB3D_SCENES:
        for scene_type in ["Egocentric", "Non-Egocentric"]:
            scene_path = os.path.join(dataset_path, scene, scene_type)
            test_file = os.path.join(scene_path, "test.txt")

            if not os.path.exists(test_file):
                continue

            with open(test_file, "r") as f:
                indices = [int(line.strip()) for line in f if line.strip()]

            dst_base = os.path.join(output_dir, scene, scene_type)

            _copy_or_link(
                os.path.join(scene_path, "test.txt"),
                os.path.join(dst_base, "test.txt"),
                symlink,
            )
            _copy_or_link(
                os.path.join(scene_path, "train.txt"),
                os.path.join(dst_base, "train.txt"),
                symlink,
            )
            for idx in indices:
                t = str(idx).zfill(5)

                _copy_or_link(
                    os.path.join(scene_path, "images", f"{t}_rgb.png"),
                    os.path.join(dst_base, "images", f"{t}_rgb.png"),
                    symlink,
                )
                _copy_or_link(
                    os.path.join(scene_path, "depths", f"{t}_depth.exr"),
                    os.path.join(dst_base, "depths", f"{t}_depth.exr"),
                    symlink,
                )
                _copy_or_link(
                    os.path.join(scene_path, "cameras", f"{t}_cam.json"),
                    os.path.join(dst_base, "cameras", f"{t}_cam.json"),
                    symlink,
                )

                # Кубмапа + depth-грани + camera JSON для каждой грани
                cam_json_path = os.path.join(scene_path, "cameras", f"{t}_cam.json")
                pano_world_pos = (
                    _read_ob3d_world_pos(cam_json_path)
                    if os.path.exists(cam_json_path)
                    else None
                )
                _export_cubemap_data(
                    pano_rgb_path=os.path.join(scene_path, "images", f"{t}_rgb.png"),
                    depth_path=os.path.join(scene_path, "depths", f"{t}_depth.exr"),
                    out_dir=os.path.join(dst_base, "data"),
                    prefix=t,
                    face_size=400,
                    pano_world_pos=pano_world_pos,
                )
                total += 1

    print(f"  Экспортировано {total} записей -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Экспорт панорам тестовой выборки с сохранением структуры папок"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["2d3ds", "structured3d", "ob3d"],
        choices=["2d3ds", "structured3d", "ob3d"],
        help="Датасеты для экспорта (default: все)",
    )
    parser.add_argument(
        "--path-2d3ds",
        default="/mnt/d/datasets/2D-3D-Semantics",
        help="Путь к 2D-3D-Semantics",
    )
    parser.add_argument(
        "--path-structured3d",
        default="/mnt/d/datasets/Structured3D",
        help="Путь к Structured3D",
    )
    parser.add_argument(
        "--path-ob3d",
        default="/mnt/d/datasets/OB3D",
        help="Путь к OB3D",
    )
    parser.add_argument(
        "--splits-path",
        default="/mnt/e/3D/experiments/splits",
        help="Директория с файлами разбиения",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/e/3D/experiments/test",
        help="Директория для экспорта (default: experiments/test)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Создавать символические ссылки вместо копирования файлов",
    )
    args = parser.parse_args()

    if "2d3ds" in args.datasets:
        print("\n=== 2D-3D-Semantics ===")
        entries = load_test_entries(args.splits_path, "2d3ds")
        if entries:
            out = os.path.join(args.output_dir, "2d3ds")
            export_2d3ds(args.path_2d3ds, entries, out, args.symlink)

    if "structured3d" in args.datasets:
        print("\n=== Structured3D ===")
        entries = load_test_entries(args.splits_path, "structured3d")
        if entries:
            out = os.path.join(args.output_dir, "structured3d")
            export_structured3d(args.path_structured3d, entries, out, args.symlink)

    if "ob3d" in args.datasets:
        print("\n=== OB3D ===")
        out = os.path.join(args.output_dir, "ob3d")
        export_ob3d(args.path_ob3d, out, args.symlink)

    print("\nГотово!")


if __name__ == "__main__":
    main()
