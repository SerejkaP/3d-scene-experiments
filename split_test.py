"""
Создание тестовой выборки панорам для 2D-3D-Semantics и Structured3D.

Для каждого датасета формируется файл test.txt со списком панорам.
Формат записей:
  - 2D-3D-Semantics: area_N/pano_filename.png
  - Structured3D:    scene_XXXXX/room_id

Скрипт сканирует директории датасетов, случайным образом отбирает
указанную долю панорам и сохраняет списки в output_dir.
"""

import argparse
import os
import random
from pathlib import Path


def collect_2d3ds_panoramas(dataset_path: str, min_cameras: int = 1) -> list[str]:
    """Собирает все панорамы из 2D-3D-Semantics, у которых есть камеры в data/pose.

    Args:
        min_cameras: Минимальное количество камер в data/pose для включения панорамы.

    Returns:
        Список строк вида "area_N/filename.png"
    """
    panoramas = []
    dataset = Path(dataset_path)
    areas = sorted(p.name for p in dataset.iterdir() if p.name.startswith("area_"))
    for area in areas:
        pano_rgb = dataset / area / "pano" / "rgb"
        data_pose = dataset / area / "data" / "pose"
        if not pano_rgb.is_dir():
            print(f"  [warn] {pano_rgb} не найдена, пропуск")
            continue
        panos = sorted(f.name for f in pano_rgb.iterdir() if f.suffix == ".png")
        kept = 0
        skipped = 0
        for pano in panos:
            # Имя панорамы без лишних суффиксов: первые 5 частей через _
            pano_name = "_".join(pano.split("_")[:5])
            if data_pose.is_dir():
                n_cameras = sum(
                    1 for f in data_pose.iterdir()
                    if f.name.startswith(f"{pano_name}_") and f.suffix == ".json"
                )
            else:
                n_cameras = 0
            if n_cameras >= min_cameras:
                panoramas.append(f"{area}/{pano}")
                kept += 1
            else:
                skipped += 1
        print(f"  {area}: {kept} панорам (пропущено {skipped} без камер)")
    return panoramas


def collect_structured3d_panoramas(dataset_path: str) -> list[str]:
    """Собирает все панорамы из Structured3D.

    Проверяет наличие rgb_rawlight.png в panorama/full/ для каждой комнаты.

    Returns:
        Список строк вида "scene_XXXXX/room_id"
    """
    panoramas = []
    dataset = Path(dataset_path)
    scenes = sorted(p.name for p in dataset.iterdir() if p.name.startswith("scene_"))
    for scene in scenes:
        rendering = dataset / scene / "2D_rendering"
        if not rendering.is_dir():
            continue
        rooms = sorted(p.name for p in rendering.iterdir() if p.is_dir())
        count = 0
        for room in rooms:
            pano_path = rendering / room / "panorama" / "full" / "rgb_rawlight.png"
            if pano_path.exists():
                panoramas.append(f"{scene}/{room}")
                count += 1
        if count > 0:
            print(f"  {scene}: {count} панорам")
    return panoramas


def make_split(
    panoramas: list[str],
    seed: int,
    test_ratio: float | None = None,
    test_count: int | None = None,
) -> tuple[list[str], list[str]]:
    """Разбивает список на train и test.

    Указывается либо test_ratio (доля), либо test_count (точное количество).
    """
    rng = random.Random(seed)
    shuffled = panoramas.copy()
    rng.shuffle(shuffled)
    if test_count is not None:
        n_test = min(test_count, len(shuffled))
    else:
        n_test = max(1, round(len(shuffled) * test_ratio))
    test = sorted(shuffled[:n_test])
    train = sorted(shuffled[n_test:])
    return train, test


def save_split(items: list[str], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(item + "\n")
    print(f"  Сохранено {len(items)} записей -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Формирование тестовой выборки панорам"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["2d3ds", "structured3d"],
        choices=["2d3ds", "structured3d"],
        help="Датасеты для обработки (default: оба)",
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
        "--output-dir",
        default="/mnt/e/3D/experiments/splits",
        help="Директория для сохранения файлов разбиения",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Доля тестовой выборки (default: 0.1 если --test-count не указан)",
    )
    group.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Точное количество панорам в тестовой выборке",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed для воспроизводимости (default: 42)",
    )
    parser.add_argument(
        "--min-cameras",
        type=int,
        default=5,
        help="Минимальное количество камер в data/pose для включения панорамы (default: 1)",
    )
    args = parser.parse_args()

    # Если ни ratio, ни count не указаны — используем ratio=0.1
    test_ratio = args.test_ratio
    test_count = args.test_count
    if test_ratio is None and test_count is None:
        test_ratio = 0.1

    if "2d3ds" in args.datasets:
        print("\n=== 2D-3D-Semantics ===")
        panoramas = collect_2d3ds_panoramas(args.path_2d3ds, min_cameras=args.min_cameras)
        print(f"Всего панорам: {len(panoramas)}")
        train, test = make_split(panoramas, args.seed, test_ratio=test_ratio, test_count=test_count)
        print(f"Train: {len(train)}, Test: {len(test)}")
        save_split(test, os.path.join(args.output_dir, "2d3ds", "test.txt"))
        save_split(train, os.path.join(args.output_dir, "2d3ds", "train.txt"))

    if "structured3d" in args.datasets:
        print("\n=== Structured3D ===")
        panoramas = collect_structured3d_panoramas(args.path_structured3d)
        print(f"Всего панорам: {len(panoramas)}")
        train, test = make_split(panoramas, args.seed, test_ratio=test_ratio, test_count=test_count)
        print(f"Train: {len(train)}, Test: {len(test)}")
        save_split(test, os.path.join(args.output_dir, "structured3d", "test.txt"))
        save_split(train, os.path.join(args.output_dir, "structured3d", "train.txt"))

    print("\nГотово!")


if __name__ == "__main__":
    main()
