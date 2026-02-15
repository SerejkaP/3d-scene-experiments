"""Утилиты для загрузки train/test разбиений датасетов."""

import os


def load_split(splits_path: str, dataset_name: str, split: str = "test") -> set[str] | None:
    """Загружает список записей из файла разбиения.

    Args:
        splits_path: Корневая директория со сплитами (например experiments/splits).
        dataset_name: Имя датасета ("2d3ds", "structured3d").
        split: Тип разбиения ("test" или "train").

    Returns:
        Множество строк из файла или None если файл не найден.
    """
    split_file = os.path.join(splits_path, dataset_name, f"{split}.txt")
    if not os.path.exists(split_file):
        return None
    with open(split_file, "r") as f:
        entries = {line.strip() for line in f if line.strip()}
    print(f"[splits] Loaded {len(entries)} entries from {split_file}")
    return entries


def filter_2d3ds_panos(pano_list: list[str], area: str, split_entries: set[str] | None) -> list[str]:
    """Фильтрует список панорам 2D-3D-Semantics по сплиту.

    Формат записей в сплите: "area_N/filename.png"
    """
    if split_entries is None:
        return pano_list
    return [p for p in pano_list if f"{area}/{p}" in split_entries]


def filter_structured3d_rooms(rooms: list[str], scene: str, split_entries: set[str] | None) -> list[str]:
    """Фильтрует список комнат Structured3D по сплиту.

    Формат записей в сплите: "scene_XXXXX/room_id"
    """
    if split_entries is None:
        return rooms
    return [r for r in rooms if f"{scene}/{r}" in split_entries]
