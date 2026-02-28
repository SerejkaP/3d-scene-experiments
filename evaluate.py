import json
import os
import hydra
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm
from utils.gsplat_utils import (
    compute_scene_radius,
    render_camera,
    render_pano,
    render_virtual_camera,
)
from utils.dataset_2d3ds_utils import pose_json_by_image_path
from utils.metrics import ClipDistanceMetric, LpipsMetric, FidMetric, DepthMetrics
from utils.tensorboard_logger import TensorBoardLogger
from utils.metrics_computer import MetricsComputer
from utils.splits import load_split, filter_2d3ds_panos

# Keys present in camera/scene metric dicts
_SCENE_KEYS = ("lpips", "clip_distance", "psnr", "ssim", "niqe", "brisque", "fid")
_DEPTH_KEYS = ("rmse", "abs_rel")


def _avg(metrics_list: list, keys: tuple) -> dict:
    """Average a list of metric dicts over the given keys, skipping NaN values."""
    result = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m and not np.isnan(float(m[k]))]
        result[k] = float(np.mean(vals)) if vals else float("nan")
    return result


def _print_and_log_averages(
    label: str,
    avg: dict,
    tb_logger: TensorBoardLogger,
    tag_prefix: str,
    step: int = 0,
) -> None:
    """Print averaged metrics and log them to TensorBoard."""
    print(f"\n{'-' * 60}")
    print(f"  {label}")
    print("-" * 60)
    for k, v in avg.items():
        if not np.isnan(v):
            print(f"  {k:20s}: {v:.4f}")
            tb_logger.log_scalar(f"{tag_prefix}/{k}", v, step)
    print("-" * 60)


def _limit_reached(counter, generation_iters):
    return generation_iters != -1 and counter >= generation_iters


def render_2d3ds(
    ply_file,
    pano_json_path,
    camera_json_path,
    output_path,
    output_depth_path=None,
    swap_yz=False,
    center_pos_scale: float = 1.0,
):
    with open(pano_json_path, "r") as f:
        pano_data = json.load(f)

    with open(camera_json_path, "r") as f:
        camera_data = json.load(f)

    pano_location = np.array(pano_data["camera_location"])
    camera_location = np.array(camera_data["camera_location"])
    if center_pos_scale != 1.0:
        camera_location = (
            pano_location + (camera_location - pano_location) * center_pos_scale
        )

    render_camera(
        ply_file,
        pano_location=pano_location,
        camera_location=camera_location,
        pano_rt=np.array(pano_data["camera_rt_matrix"]),
        camera_rt=np.array(camera_data["camera_rt_matrix"]),
        camera_k=np.array(camera_data["camera_k_matrix"]),
        output_path=output_path,
        output_depth_path=output_depth_path,
        swap_yz=swap_yz,
    )


def evaluate_2d3ds(
    cfg,
    save_path,
    tb_logger: TensorBoardLogger,
    swap_yz: bool = False,
    restricted_data: list[str] = [],
) -> dict:
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
    dataset_path = str(cfg.dataset.path)
    # Получение выборки, по которой будут считаться метрики
    split_entries = load_split(str(cfg.splits_path), "2d3ds")
    areas = sorted([s for s in os.listdir(dataset_path) if s.startswith("area_")])

    all_scene_metrics = []
    all_scene_depth_metrics = []

    counter = 0
    for area in areas:
        if _limit_reached(counter, cfg.generation_iters):
            break
        pano_images = os.path.join(dataset_path, area, "pano", "rgb")
        pano_pose = os.path.join(dataset_path, area, "pano", "pose")
        data_pose = os.path.join(dataset_path, area, "data", "pose")
        data_images = os.path.join(dataset_path, area, "data", "rgb")
        data_depth = os.path.join(dataset_path, area, "data", "depth")
        pano_img_list = sorted(
            [img for img in os.listdir(pano_images) if img.endswith(".png")]
        )
        # Получить только тестовые панорамы (поиск панорам по выборке)
        pano_img_list = filter_2d3ds_panos(pano_img_list, area, split_entries)
        if len(pano_img_list) == 0:
            continue

        for current_pano_rgb_name in pano_img_list:
            if _limit_reached(counter, cfg.generation_iters):
                break

            scene_name = os.path.splitext(current_pano_rgb_name)[0]
            current_pano_rgb_path = os.path.join(pano_images, current_pano_rgb_name)
            area_save_path = os.path.join(save_path, area)
            ply_render_path = os.path.join(area_save_path, f"{scene_name}_render.ply")

            if not os.path.exists(ply_render_path):
                print(f"[2d3ds] PLY not found, skipping: {ply_render_path}")
                continue

            # По названию панорамы получаю json с информацией о позиции камеры
            pano_json_path, pano_name = pose_json_by_image_path(
                current_pano_rgb_name, pano_pose
            )
            # Куда будет сохранена панорама из сцены и карта глубины панорамы
            rendered_pano_path = os.path.join(area_save_path, f"{scene_name}_pano.png")
            rendered_pano_depth_path = os.path.join(
                area_save_path, f"{scene_name}_depth.exr"
            )
            # Список файлов, которые будут удалены в конце сбора метрик для сцены
            files_to_remove = []
            is_restricted_file = current_pano_rgb_path in restricted_data

            # Если необходимо удалять рендеры и сцена не в списке неудаляемых, то удалить рендеры
            if cfg.remove_renders and not is_restricted_file:
                files_to_remove.append(rendered_pano_path)
                files_to_remove.append(rendered_pano_depth_path)

            print(f"[2d3ds] Evaluate scene for {current_pano_rgb_name}")

            # Панорамы рендерятся для визуального сравнения
            # Рендер панорамы сгенерированной сцены
            render_pano(
                ply_render_path,
                [0, 0, 0],
                cfg.dataset.pano_width,
                rendered_pano_path,
                rendered_pano_depth_path,
                swap_yz=swap_yz,
            )

            # Возможно, стоит убрать. Логирование картинки панорамы
            metrics_computer.log_image_comparison(
                tag=f"Images/Panorama/{scene_name}",
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
            camera_depth_metrics_list = []
            gt_paths_for_fid = []
            render_paths_for_fid = []

            # Авто-масштабирование позиций камер под координатное пространство GS-сцены.
            # Необходимо для DreamScene360 (swap_yz=True): GS генерируется в нормализованном
            # масштабе, а позиции 2d3ds — в метрах. Масштаб = scene_radius / max_cam_offset.
            center_pos_scale = 1.0
            if swap_yz and os.path.exists(ply_render_path):
                scene_radius = compute_scene_radius(ply_render_path)
                with open(pano_json_path) as _f:
                    _pano_loc = np.array(json.load(_f)["camera_location"])
                _max_cam_dist = 0.0
                for _cp in camera_poses:
                    with open(os.path.join(data_pose, _cp)) as _f:
                        _cam_loc = np.array(json.load(_f)["camera_location"])
                    _max_cam_dist = max(
                        _max_cam_dist, float(np.linalg.norm(_cam_loc - _pano_loc))
                    )
                if _max_cam_dist > 1e-6:
                    center_pos_scale = scene_radius / _max_cam_dist
                    print(
                        f"  [auto-scale] scene_radius={scene_radius:.3f}, "
                        f"max_cam_dist={_max_cam_dist:.3f}, "
                        f"center_pos_scale={center_pos_scale:.4f}"
                    )

            for camera_idx, camera_pose in enumerate(tqdm(camera_poses)):
                camera_json_path = os.path.join(data_pose, camera_pose)
                rendered_camera_subname = "_".join(camera_pose.split("_")[:-1])
                rendered_camera = f"{rendered_camera_subname}_render.png"
                # Путь сохранения получивегося изображения камеры
                rendered_camera_path = os.path.join(area_save_path, rendered_camera)
                # Путь сохранения получившейся глубины камеры
                rendered_camera_depth_path = os.path.join(
                    area_save_path, f"{rendered_camera_subname}_depth.exr"
                )
                # Если необходимо удалять рендеры и сцена не в списке неудаляемых, то удалить рендеры
                if cfg.remove_renders and not is_restricted_file:
                    files_to_remove.append(rendered_camera_path)
                    files_to_remove.append(rendered_camera_depth_path)

                # Рендер камеры в сцене в заданной позиции
                render_2d3ds(
                    ply_render_path,
                    pano_json_path,
                    camera_json_path,
                    rendered_camera_path,
                    rendered_camera_depth_path,
                    swap_yz=swap_yz,
                    center_pos_scale=center_pos_scale,
                )

                # Пути к реальному изображению и карте глубины камеры
                gt_image_path = os.path.join(
                    data_images, f"{rendered_camera_subname}_rgb.png"
                )
                gt_depth_path = os.path.join(
                    data_depth, f"{rendered_camera_subname}_depth.png"
                )

                # Метрики сравения двух изображений между собой (PSNR, SSIM, CLIP-Dist, LPIPS, NIQE, BRISQUE)
                camera_metrics = metrics_computer.compute_camera_metrics(
                    gt_image_path=gt_image_path,
                    rendered_image_path=rendered_camera_path,
                )
                # Логирование получившихся метрик для камеры
                metrics_computer.log_camera_metrics(
                    metrics_dict=camera_metrics,
                    scene_name=pano_name,
                    camera_idx=camera_idx,
                )
                # Добавление метрик изображения в общий список для усреднения
                camera_metrics_list.append(camera_metrics)

                if os.path.exists(gt_depth_path) and os.path.exists(
                    rendered_camera_depth_path
                ):
                    # Подсчет метрик глубины для камеры
                    camera_depth_metrics = (
                        metrics_computer.compute_camera_depth_metrics(
                            gt_depth_path=gt_depth_path,
                            rendered_depth_path=rendered_camera_depth_path,
                            gt_depth_scale=1.0 / 512.0,
                        )
                    )
                    if camera_depth_metrics:
                        # Логирование метрик глубины для камеры
                        metrics_computer.log_camera_depth_metrics(
                            depth_metrics=camera_depth_metrics,
                            scene_name=pano_name,
                            camera_idx=camera_idx,
                        )
                        camera_depth_metrics_list.append(camera_depth_metrics)

                # Логирование изображения, если не набран предел по cfg.tensorboard.max_images_per_scene
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

                # Добавление пути реальной камеры для подсчета FID (считается по всем камерам для сцены)
                gt_paths_for_fid.append(gt_image_path)
                # Добавление пути отрендеренной камеры для подсчета FID (считается по всем камерам для сцены)
                render_paths_for_fid.append(rendered_camera_path)

            scene_m = metrics_computer.aggregate_and_log_scene_metrics(
                metrics_list=camera_metrics_list,
                gt_paths=gt_paths_for_fid,
                rendered_paths=render_paths_for_fid,
                step=counter,
            )
            if scene_m:
                # normalize _mean suffix → bare key for uniform aggregation
                all_scene_metrics.append(
                    {k.replace("_mean", ""): v for k, v in scene_m.items()}
                )

            if len(camera_depth_metrics_list) > 0:
                scene_rmse = sum(m["rmse"] for m in camera_depth_metrics_list) / len(
                    camera_depth_metrics_list
                )
                scene_abs_rel = sum(
                    m["abs_rel"] for m in camera_depth_metrics_list
                ) / len(camera_depth_metrics_list)

                tb_logger.log_scalar(
                    "Metrics/Scene/Depth_RMSE_mean", scene_rmse, counter
                )
                tb_logger.log_scalar(
                    "Metrics/Scene/Depth_AbsRel_mean", scene_abs_rel, counter
                )

                print(f"Scene average Depth RMSE: {scene_rmse:.4f}")
                print(f"Scene average Depth Abs Rel: {scene_abs_rel:.4f}")
                all_scene_depth_metrics.append(
                    {"rmse": scene_rmse, "abs_rel": scene_abs_rel}
                )

            for file_to_remove_path in files_to_remove:
                # Удаление всех файлов, которые не помечены restricted
                os.remove(file_to_remove_path)

            # Следующая сцена -> увеличить счетчик
            counter += 1

    return _finalize_dataset_averages(
        dataset_name="2d3ds",
        all_scene_metrics=all_scene_metrics,
        all_scene_depth_metrics=all_scene_depth_metrics,
        tb_logger=tb_logger,
    )


# Максимальная глубина в каждой сцене
MAX_DEPTH_SCENE_DICT = {
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
    "sun-temple": 20,
    "pavillion": 50,
}


def evaluate_ob3d(
    cfg,
    save_path,
    tb_logger: TensorBoardLogger,
    swap_yz: bool = False,
    restricted_data: list[str] = [],
) -> dict:
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

    all_scene_metrics = []
    all_scene_depth_metrics = []

    # Только indoor сцены
    avail_scenes = [
        "archiviz-flat",
        "barbershop",
        "classroom",
        "restroom",
        "san-miguel",
        "sun-temple",
    ]

    dataset_path = str(cfg.dataset.path)
    counter = 0
    for scene in avail_scenes:
        if _limit_reached(counter, cfg.generation_iters):
            break
        depth_metric = DepthMetrics(
            min_depth=0.0, max_depth=MAX_DEPTH_SCENE_DICT[scene]
        )
        metrics_computer.depth_metric = depth_metric
        # Оставлен подсчет метрик только для Non-Egocentric
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
            # Получение информации о тестовой выборке
            with open(test_file, "r") as f:
                test_data = [int(line.strip()) for line in f.readlines()]

            save_scene_path = os.path.join(save_path, scene, scene_type)

            for t in test_data:
                if _limit_reached(counter, cfg.generation_iters):
                    break
                t_scene = f"{str(t).zfill(5)}"
                ply_render_path = os.path.join(save_scene_path, f"{t_scene}_render.ply")

                if not os.path.exists(ply_render_path):
                    print(f"[ob3d] PLY not found, skipping: {ply_render_path}")
                    continue

                rendered_pano_path = os.path.join(
                    save_scene_path, f"{t_scene}_render_pano.png"
                )
                rendered_pano_depth_path = os.path.join(
                    save_scene_path, f"{t_scene}_render_depth.exr"
                )
                pano_rgb_path = os.path.join(images_path, f"{t_scene}_rgb.png")
                room_name = f"{scene}/{scene_type}/{t_scene}"

                # Список файлов, которые будут удалены в конце сбора метрик для сцены
                files_to_remove = []
                is_restricted_file = pano_rgb_path in restricted_data
                # Если необходимо удалять рендеры и сцена не в списке неудаляемых, то удалить рендеры
                if cfg.remove_renders and not is_restricted_file:
                    files_to_remove.append(rendered_pano_path)
                    files_to_remove.append(rendered_pano_depth_path)

                print(f"[ob3d] Evaluate scene for {room_name}")

                # Рендеринг панорамы для визуального сравнения с GT-панорамой
                render_pano(
                    ply_render_path,
                    [0, 0, 0],
                    cfg.dataset.pano_width,
                    rendered_pano_path,
                    rendered_pano_depth_path,
                    swap_yz=swap_yz,
                )

                # Возможно, стоит убрать. Логирование картинки панорамы
                metrics_computer.log_image_comparison(
                    tag=f"Images/Panorama/{room_name}",
                    gt_path=pano_rgb_path,
                    rendered_path=rendered_pano_path,
                    step=counter,
                    log_images=cfg.tensorboard.log_images,
                )

                # --- метрики кубмапа по всем тестовым панорамам сцены ---
                # Читаем экстринсики w2c текущей панорамы — они задают систему координат GS-сцены.
                # Нужны для перевода мировых позиций других панорам в GS-фрейм:
                #   center_pos = R_w2c_t @ pano_world_pos_t2 + t_w2c_t
                cameras_dir = os.path.join(scene_path, "cameras")
                cur_cam_json = os.path.join(cameras_dir, f"{t_scene}_cam.json")
                if os.path.exists(cur_cam_json):
                    # Загружаем матрицу вращения и вектор сдвига текущей панорамы
                    with open(cur_cam_json) as _fh:
                        _d = json.load(_fh)[0]
                    R_w2c_cur = np.array(_d["extrinsics"]["rotation"])
                    t_w2c_cur = np.array(_d["extrinsics"]["translation"])
                else:
                    # Файл не найден — считаем, что источник совпадает с началом координат
                    R_w2c_cur = np.eye(3)
                    t_w2c_cur = np.zeros(3)

                # Директория с тестовыми камерами (JSON + GT-изображения)
                data_dir = os.path.join(scene_path, "data")
                if os.path.isdir(data_dir):
                    # Собираем отсортированный список JSON-файлов камер
                    cam_jsons = sorted(
                        f for f in os.listdir(data_dir) if f.endswith(".json")
                    )

                    # Списки метрик и путей, накапливаемые по всем камерам сцены
                    camera_metrics_list = []
                    camera_depth_metrics_list = []
                    gt_paths_for_fid = []
                    render_paths_for_fid = []
                    # Позиция каждой панорамы в GS-фрейме (нужна для рендера полной панорамы)
                    center_pos_by_pano: dict[str, list] = {}
                    # Кэш мировых позиций панорам, чтобы не читать JSON повторно
                    _world_pos_cache: dict[str, np.ndarray] = {}
                    # Кэш матриц вращения панорам (для корректировки направления граней)
                    _rotation_cache: dict[str, np.ndarray] = {}
                    # Соответствие названий граней кубмапа ключам py360convert
                    _FACE_TO_CUBE_KEY = {
                        "front": "F",
                        "right": "R",
                        "back": "B",
                        "left": "L",
                        "top": "U",
                        "bottom": "D",
                    }

                    # Авто-масштабирование center_pos под координатное пространство GS-сцены.
                    # Необходимо для моделей вроде DreamScene360 (swap_yz=True), у которых GS
                    # генерируется в нормализованном масштабе, а позиции датасета — в метрах.
                    center_pos_scale = 1.0
                    if swap_yz and os.path.exists(ply_render_path):
                        # Вычисляем радиус сцены как 90-й перцентиль расстояний гауссиан от начала координат
                        scene_radius = compute_scene_radius(ply_render_path)
                        # Пре-пасс: находим максимальное расстояние между панорамами в GS-фрейме
                        _seen_pids: set = set()
                        _max_world_dist = 0.0
                        for _cj in cam_jsons:
                            _st = _cj[:-5]
                            _pp = _st.rsplit("_", 1)
                            # Определяем pano_id из имени файла (убираем суффикс грани)
                            _pid = (
                                _pp[0]
                                if len(_pp) == 2 and _pp[1] in _FACE_TO_CUBE_KEY
                                else _st
                            )
                            # Пропускаем уже обработанные панорамы
                            if _pid in _seen_pids:
                                continue
                            _seen_pids.add(_pid)
                            _pc = os.path.join(cameras_dir, f"{_pid}_cam.json")
                            if os.path.exists(_pc):
                                with open(_pc) as _f:
                                    _pd = json.load(_f)[0]
                                _R = np.array(_pd["extrinsics"]["rotation"])
                                _t = np.array(_pd["extrinsics"]["translation"])
                                # Мировая позиция панорамы: p_world = -R^T @ t
                                _wpos = -_R.T @ _t
                                # Позиция в GS-фрейме (системе координат source-панорамы)
                                _cp = R_w2c_cur @ _wpos + t_w2c_cur
                                # Обновляем максимальное расстояние
                                _max_world_dist = max(
                                    _max_world_dist, float(np.linalg.norm(_cp))
                                )
                        if _max_world_dist > 1e-6:
                            # Коэффициент масштаба: самая дальняя камера окажется на границе сцены
                            center_pos_scale = scene_radius / _max_world_dist
                            print(
                                f"  [auto-scale] scene_radius={scene_radius:.3f}, "
                                f"max_world_dist={_max_world_dist:.3f}, "
                                f"center_pos_scale={center_pos_scale:.4f}"
                            )

                    for cam_idx, cam_json_name in enumerate(tqdm(cam_jsons)):
                        # Имя файла без расширения используется как идентификатор кадра
                        stem = cam_json_name[:-5]  # убираем .json
                        gt_face_path = os.path.join(data_dir, f"{stem}.png")
                        # Пропускаем камеры без GT-изображения
                        if not os.path.exists(gt_face_path):
                            continue

                        # Читаем параметры камеры: интринсики, экстринсики, разрешение
                        with open(os.path.join(data_dir, cam_json_name)) as fh:
                            cam_info = json.load(fh)

                        cam = {
                            "K": np.array(cam_info["K"]),
                            "R": np.array(cam_info["R"]),
                            "width": cam_info["width"],
                            "height": cam_info["height"],
                        }

                        # Определяем, к какой панораме относится данная грань кубмапа.
                        # Имена файлов имеют формат {pano_id}_{face_name}.json
                        _parts = stem.rsplit("_", 1)
                        pano_id = (
                            _parts[0]
                            if len(_parts) == 2 and _parts[1] in _FACE_TO_CUBE_KEY
                            else stem
                        )
                        # Загружаем мировую позицию и ориентацию панорамы из директории cameras/
                        if pano_id not in _world_pos_cache:
                            _pc = os.path.join(cameras_dir, f"{pano_id}_cam.json")
                            if os.path.exists(_pc):
                                with open(_pc) as _fh:
                                    _pd = json.load(_fh)[0]
                                _R_tgt = np.array(_pd["extrinsics"]["rotation"])
                                _t_tgt = np.array(_pd["extrinsics"]["translation"])
                                # Мировая позиция: p_world = -R^T @ t
                                _world_pos_cache[pano_id] = -_R_tgt.T @ _t_tgt
                                # Сохраняем вращение для последующей коррекции направления камеры
                                _rotation_cache[pano_id] = _R_tgt
                            else:
                                # Файл не найден — панорама в начале координат с единичным вращением
                                _world_pos_cache[pano_id] = np.zeros(3)
                                _rotation_cache[pano_id] = np.eye(3)
                        world_pos_tgt = _world_pos_cache[pano_id]
                        # Переводим мировую позицию в GS-фрейм и применяем масштабный коэффициент
                        center_pos = (
                            (R_w2c_cur @ world_pos_tgt + t_w2c_cur) * center_pos_scale
                        ).tolist()

                        # Запоминаем позицию первой грани панорамы для последующего рендера полной панорамы
                        if len(_parts) == 2 and _parts[1] in _FACE_TO_CUBE_KEY:
                            center_pos_by_pano.setdefault(_parts[0], center_pos)

                        # Пути для сохранения рендера и карты глубины
                        rendered_cam_path = os.path.join(
                            save_scene_path, f"data_{stem}_render.png"
                        )
                        rendered_depth_path = os.path.join(
                            save_scene_path, f"data_{stem}_depth.exr"
                        )
                        # Если необходимо удалять рендеры и сцена не в списке неудаляемых, то удалить рендеры
                        if cfg.remove_renders and not is_restricted_file:
                            files_to_remove.append(rendered_cam_path)
                            files_to_remove.append(rendered_depth_path)

                        gt_depth_path = os.path.join(data_dir, f"{stem}_depth.exr")

                        # Корректируем матрицу вращения камеры: cam["R"] задаёт направление грани
                        # в локальном фрейме панорамы pano_id, но GS-сцена живёт в фрейме t_scene.
                        # Цепочка преобразований: GS-фрейм → мировой → фрейм pano_id → фрейм камеры.
                        # R_corrected = cam["R"] @ R_w2c_tgt @ R_w2c_cur.T
                        R_w2c_tgt = _rotation_cache.get(pano_id, np.eye(3))
                        cam_for_render = dict(cam)
                        cam_for_render["R"] = cam["R"] @ R_w2c_tgt @ R_w2c_cur.T

                        # Рендерим грань из GS-сцены с исправленными позицией и ориентацией
                        render_virtual_camera(
                            ply_path=ply_render_path,
                            camera=cam_for_render,
                            output_path=rendered_cam_path,
                            output_depth_path=rendered_depth_path,
                            center_pos=center_pos,
                            swap_yz=swap_yz,
                        )

                        # Вычисляем метрики качества (LPIPS, PSNR, SSIM, CLIP) для данной грани
                        camera_metrics = metrics_computer.compute_camera_metrics(
                            gt_image_path=gt_face_path,
                            rendered_image_path=rendered_cam_path,
                        )
                        # Логируем метрики в TensorBoard с привязкой к сцене и индексу камеры
                        metrics_computer.log_camera_metrics(
                            metrics_dict=camera_metrics,
                            scene_name=room_name,
                            camera_idx=cam_idx,
                        )
                        # Логируем сравнительные изображения GT / рендер (с ограничением по количеству)
                        metrics_computer.log_image_comparison(
                            tag=f"Images/Camera/{room_name}/{stem}",
                            gt_path=gt_face_path,
                            rendered_path=rendered_cam_path,
                            step=counter,
                            log_images=(
                                cfg.tensorboard.log_images
                                and cam_idx < cfg.tensorboard.max_images_per_scene
                            ),
                        )

                        # Вычисляем метрики глубины, если оба файла (GT и рендер) существуют
                        if os.path.exists(gt_depth_path) and os.path.exists(
                            rendered_depth_path
                        ):
                            camera_depth_metrics = (
                                metrics_computer.compute_camera_depth_metrics(
                                    gt_depth_path=gt_depth_path,
                                    rendered_depth_path=rendered_depth_path,
                                    gt_depth_scale=1.0,
                                )
                            )
                            if camera_depth_metrics:
                                # Логируем метрики глубины и добавляем в список для усреднения по сцене
                                metrics_computer.log_camera_depth_metrics(
                                    depth_metrics=camera_depth_metrics,
                                    scene_name=room_name,
                                    camera_idx=cam_idx,
                                )
                                camera_depth_metrics_list.append(camera_depth_metrics)

                        # Накапливаем метрики и пути для агрегации по сцене и расчёта FID
                        camera_metrics_list.append(camera_metrics)
                        gt_paths_for_fid.append(gt_face_path)
                        render_paths_for_fid.append(rendered_cam_path)

                    # Агрегируем метрики по всем камерам сцены и логируем средние значения
                    if camera_metrics_list:
                        scene_m = metrics_computer.aggregate_and_log_scene_metrics(
                            metrics_list=camera_metrics_list,
                            gt_paths=gt_paths_for_fid,
                            rendered_paths=render_paths_for_fid,
                            step=counter,
                        )
                        if scene_m:
                            # Убираем суффикс "_mean" из ключей для единообразия
                            all_scene_metrics.append(
                                {k.replace("_mean", ""): v for k, v in scene_m.items()}
                            )

                    # Усредняем метрики глубины по сцене и логируем
                    if camera_depth_metrics_list:
                        scene_rmse = sum(
                            m["rmse"] for m in camera_depth_metrics_list
                        ) / len(camera_depth_metrics_list)
                        scene_abs_rel = sum(
                            m["abs_rel"] for m in camera_depth_metrics_list
                        ) / len(camera_depth_metrics_list)
                        tb_logger.log_scalar(
                            "Metrics/Scene/Depth_RMSE_mean", scene_rmse, counter
                        )
                        tb_logger.log_scalar(
                            "Metrics/Scene/Depth_AbsRel_mean", scene_abs_rel, counter
                        )
                        print(f"Scene average Depth RMSE: {scene_rmse:.4f}")
                        print(f"Scene average Depth Abs Rel: {scene_abs_rel:.4f}")
                        all_scene_depth_metrics.append(
                            {"rmse": scene_rmse, "abs_rel": scene_abs_rel}
                        )

                for file_to_remove_path in files_to_remove:
                    # Удаление всех файлов, которые не помечены restricted
                    os.remove(file_to_remove_path)

                counter += 1

    return _finalize_dataset_averages(
        dataset_name="ob3d",
        all_scene_metrics=all_scene_metrics,
        all_scene_depth_metrics=all_scene_depth_metrics,
        tb_logger=tb_logger,
    )


def _finalize_dataset_averages(
    dataset_name: str,
    all_scene_metrics: list,
    all_scene_depth_metrics: list,
    tb_logger: TensorBoardLogger,
) -> dict:
    """Compute, print, and TensorBoard-log dataset-level averages. Return them."""
    result = {}

    if all_scene_metrics:
        avg_scene = _avg(all_scene_metrics, _SCENE_KEYS)
        _print_and_log_averages(
            label=f"[{dataset_name}] Dataset average — Scene/Camera ({len(all_scene_metrics)} scenes)",
            avg=avg_scene,
            tb_logger=tb_logger,
            tag_prefix=f"DatasetAvg/{dataset_name}/Scene",
        )
        result["scene"] = avg_scene

    if all_scene_depth_metrics:
        avg_scene_depth = _avg(all_scene_depth_metrics, _DEPTH_KEYS)
        _print_and_log_averages(
            label=f"[{dataset_name}] Dataset average — Scene Depth ({len(all_scene_depth_metrics)} scenes)",
            avg=avg_scene_depth,
            tb_logger=tb_logger,
            tag_prefix=f"DatasetAvg/{dataset_name}/SceneDepth",
        )
        result["scene_depth"] = avg_scene_depth

    return result


@hydra.main(config_path="conf", config_name="config", version_base=None)
def evaluate(cfg: DictConfig):
    tb_logger = TensorBoardLogger(
        log_dir=cfg.tensorboard.log_dir,
        enabled=cfg.tensorboard.enabled,
        flush_secs=cfg.tensorboard.flush_secs,
        max_image_size=2048,
    )

    dataset_name = str(cfg.dataset.name)
    print(f"\n{'='*60}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'='*60}\n")

    save_path = os.path.join(cfg.save_path, cfg.model.name, dataset_name)

    if not os.path.isdir(save_path):
        raise Exception(f"Output directory not found!")

    swap_yz = bool(cfg.model.get("swap_yz", False))

    restricted_data = []
    if cfg.dataset.restricted_data_path:
        with open(str(cfg.dataset.restricted_data_path), "r") as _rf:
            restricted_data = _rf.readlines()

    if dataset_name == "2d3ds":
        evaluate_2d3ds(cfg, save_path, tb_logger, swap_yz, restricted_data)
    elif dataset_name == "ob3d":
        evaluate_ob3d(cfg, save_path, tb_logger, swap_yz, restricted_data)
    else:
        raise Exception(f"Unknown dataset: {dataset_name}!")

    tb_logger.close()


if __name__ == "__main__":
    evaluate()
