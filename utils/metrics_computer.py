from logging import info
import numpy as np
from typing import Optional, Dict, List
from utils.metrics import (
    LpipsMetric,
    ClipDistanceMetric,
    FidMetric,
    DepthMetrics,
    compute_gt_metrics,
    compute_niqe,
    compute_brisque,
)
from utils.tensorboard_logger import TensorBoardLogger
from utils.gsplat_utils import load_depth


class MetricsComputer:
    """
    Unified metrics computation, logging, and printing for panorama and camera views.
    Eliminates code duplication across different datasets (2d3ds, ob3d, structured3d).
    """

    def __init__(
        self,
        lpips_metric: LpipsMetric,
        clip_metric: ClipDistanceMetric,
        fid_metric: FidMetric,
        depth_metric: Optional[DepthMetrics],
        tb_logger: TensorBoardLogger,
    ):
        """
        Initialize MetricsComputer with pre-initialized metric instances.

        Args:
            lpips_metric: LPIPS perceptual metric instance
            clip_metric: CLIP distance metric instance
            fid_metric: FID metric instance
            depth_metric: Optional depth metrics instance (None for datasets without depth)
            tb_logger: TensorBoard logger instance
        """
        self.lpips_metric = lpips_metric
        self.clip_metric = clip_metric
        self.fid_metric = fid_metric
        self.depth_metric = depth_metric
        self.tb_logger = tb_logger

    def compute_and_log_panorama_metrics(
        self,
        gt_path: str,
        rendered_path: str,
        step: int,
        scene_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute all panorama metrics (LPIPS, CLIP, PSNR, SSIM, NIQE, BRISQUE),
        log them to TensorBoard, and print to console.

        Args:
            gt_path: Path to ground truth panorama image
            rendered_path: Path to rendered panorama image
            step: Global step for TensorBoard logging
            scene_name: Optional scene name for logging tags

        Returns:
            Dictionary with metric values: {lpips, clip_distance, psnr, ssim, niqe, brisque}
        """
        # Compute reference-based metrics
        pano_lpips = self.lpips_metric.compute(gt_path, rendered_path)
        pano_clip_d = self.clip_metric.compute(gt_path, rendered_path)
        pano_psnr, pano_ssim = compute_gt_metrics(gt_path, rendered_path)

        # Compute reference-free metrics
        pano_niqe = compute_niqe(rendered_path)
        pano_brisque = compute_brisque(rendered_path)

        # Log to TensorBoard
        self.tb_logger.log_scalar("Metrics/Panorama/LPIPS", pano_lpips, step)
        self.tb_logger.log_scalar("Metrics/Panorama/CLIP-Distance", pano_clip_d, step)
        self.tb_logger.log_scalar("Metrics/Panorama/PSNR", pano_psnr, step)
        self.tb_logger.log_scalar("Metrics/Panorama/SSIM", pano_ssim, step)
        self.tb_logger.log_scalar("Metrics/Panorama/NIQE", pano_niqe, step)
        self.tb_logger.log_scalar("Metrics/Panorama/BRISQUE", pano_brisque, step)

        # Print to console
        info(f"PANO LPIPS: {pano_lpips}")
        info(f"PANO CLIP-Distance: {pano_clip_d}")
        info(f"PANO PSNR: {pano_psnr}")
        info(f"PANO SSIM: {pano_ssim}")
        info(f"PANO NIQE: {pano_niqe}")
        info(f"PANO BRISQUE: {pano_brisque}")

        return {
            "lpips": pano_lpips,
            "clip_distance": pano_clip_d,
            "psnr": pano_psnr,
            "ssim": pano_ssim,
            "niqe": pano_niqe,
            "brisque": pano_brisque,
        }

    def compute_and_log_panorama_depth_metrics(
        self,
        gt_depth_path: str,
        rendered_depth_path: str,
        step: int,
        scene_name: str,
        gt_depth_scale: float = 1.0 / 1000.0,
    ) -> Optional[Dict[str, float]]:
        """
        Compute panorama depth metrics (RMSE, AbsRel) with error handling.

        Args:
            gt_depth_path: Path to ground truth depth map
            rendered_depth_path: Path to rendered depth map
            step: Global step for TensorBoard logging
            scene_name: Scene name for error messages
            gt_depth_scale: Scale factor for ground truth depth (default: 1.0/1000.0 for mm->m)

        Returns:
            Dictionary with {rmse, abs_rel} or None if computation fails
        """
        if not self.depth_metric:
            return None

        try:
            # Load depth maps
            gt_depth = load_depth(gt_depth_path, depth_scale=gt_depth_scale)
            render_depth = load_depth(
                rendered_depth_path,
                depth_scale=(
                    1.0 / 1000.0 if rendered_depth_path.endswith(".png") else 1.0
                ),
            )

            # Compute depth metrics
            depth_metrics = self.depth_metric.compute(gt_depth, render_depth)

            if not np.isnan(depth_metrics["rmse"]):
                rmse = depth_metrics["rmse"]
                abs_rel = depth_metrics["abs_rel"]

                # Log to TensorBoard
                self.tb_logger.log_scalar("Metrics/Panorama/Depth_RMSE", rmse, step)
                self.tb_logger.log_scalar(
                    "Metrics/Panorama/Depth_AbsRel", abs_rel, step
                )

                # Print to console
                info(f"PANO Depth RMSE: {rmse:.4f}")
                info(f"PANO Depth Abs Rel: {abs_rel:.4f}")

                return {"rmse": rmse, "abs_rel": abs_rel}

        except Exception as e:
            print(
                f"Warning: Could not compute panorama depth metrics for {scene_name}: {e}"
            )

        return None

    def compute_camera_metrics(
        self, gt_image_path: str, rendered_image_path: str
    ) -> Dict[str, float]:
        """
        Compute metrics for a single perspective camera view.

        Args:
            gt_image_path: Path to ground truth image
            rendered_image_path: Path to rendered image

        Returns:
            Dictionary with metric values: {lpips, clip_distance, psnr, ssim, niqe, brisque}
        """
        # Reference-based metrics
        lpips = self.lpips_metric.compute(gt_image_path, rendered_image_path)
        clip_distance = self.clip_metric.compute(gt_image_path, rendered_image_path)
        psnr, ssim = compute_gt_metrics(gt_image_path, rendered_image_path)

        # Reference-free metrics
        niqe = compute_niqe(rendered_image_path)
        brisque = compute_brisque(rendered_image_path)

        return {
            "lpips": lpips,
            "clip_distance": clip_distance,
            "psnr": psnr,
            "ssim": ssim,
            "niqe": niqe,
            "brisque": brisque,
        }

    def log_camera_metrics(
        self, metrics_dict: Dict[str, float], scene_name: str, camera_idx: int
    ) -> None:
        """
        Log camera metrics to TensorBoard.

        Args:
            metrics_dict: Dictionary with metric values from compute_camera_metrics()
            scene_name: Scene name for TensorBoard tag
            camera_idx: Camera index for TensorBoard step
        """
        self.tb_logger.log_scalar(
            f"Metrics/Camera_{scene_name}/LPIPS", metrics_dict["lpips"], camera_idx
        )
        self.tb_logger.log_scalar(
            f"Metrics/Camera_{scene_name}/CLIP-Distance",
            metrics_dict["clip_distance"],
            camera_idx,
        )
        self.tb_logger.log_scalar(
            f"Metrics/Camera_{scene_name}/PSNR", metrics_dict["psnr"], camera_idx
        )
        self.tb_logger.log_scalar(
            f"Metrics/Camera_{scene_name}/SSIM", metrics_dict["ssim"], camera_idx
        )
        self.tb_logger.log_scalar(
            f"Metrics/Camera_{scene_name}/NIQE", metrics_dict["niqe"], camera_idx
        )
        self.tb_logger.log_scalar(
            f"Metrics/Camera_{scene_name}/BRISQUE", metrics_dict["brisque"], camera_idx
        )

    def compute_camera_depth_metrics(
        self,
        gt_depth_path: str,
        rendered_depth_path: str,
        gt_depth_scale: float = 1.0 / 1000.0,
    ) -> Optional[Dict[str, float]]:
        """
        Compute depth metrics for a perspective camera view.

        Args:
            gt_depth_path: Path to ground truth depth map
            rendered_depth_path: Path to rendered depth map
            gt_depth_scale: Scale factor for ground truth depth

        Returns:
            Dictionary with {rmse, abs_rel} or None if computation fails
        """
        if not self.depth_metric:
            return None

        try:
            gt_depth = load_depth(gt_depth_path, depth_scale=gt_depth_scale)
            render_depth = load_depth(
                rendered_depth_path,
                depth_scale=(
                    1.0 / 1000.0 if rendered_depth_path.endswith(".png") else 1.0
                ),
            )

            depth_metrics = self.depth_metric.compute(gt_depth, render_depth)

            if not np.isnan(depth_metrics["rmse"]):
                return {
                    "rmse": depth_metrics["rmse"],
                    "abs_rel": depth_metrics["abs_rel"],
                }
        except Exception as e:
            print(f"Warning: Could not compute camera depth metrics: {e}")

        return None

    def log_camera_depth_metrics(
        self, depth_metrics: Dict[str, float], scene_name: str, camera_idx: int
    ) -> None:
        """
        Log camera depth metrics to TensorBoard.

        Args:
            depth_metrics: Dictionary with {rmse, abs_rel}
            scene_name: Scene name for TensorBoard tag
            camera_idx: Camera index for TensorBoard step
        """
        self.tb_logger.log_scalar(
            f"Metrics/Camera_{scene_name}/Depth_RMSE",
            depth_metrics["rmse"],
            camera_idx,
        )
        self.tb_logger.log_scalar(
            f"Metrics/Camera_{scene_name}/Depth_AbsRel",
            depth_metrics["abs_rel"],
            camera_idx,
        )

    def aggregate_and_log_scene_metrics(
        self,
        metrics_list: List[Dict[str, float]],
        gt_paths: List[str],
        rendered_paths: List[str],
        step: int,
    ) -> Dict[str, float]:
        """
        Aggregate camera metrics to scene-level averages, compute FID, log and print.

        Args:
            metrics_list: List of metric dictionaries from compute_camera_metrics()
            gt_paths: List of ground truth image paths for FID computation
            rendered_paths: List of rendered image paths for FID computation
            step: Global step for TensorBoard logging

        Returns:
            Dictionary with scene-level metrics
        """
        num_views = len(metrics_list)
        if num_views == 0:
            return {}

        # Aggregate metrics
        scene_lpips = sum(m["lpips"] for m in metrics_list) / num_views
        scene_clip_d = sum(m["clip_distance"] for m in metrics_list) / num_views
        scene_psnr = sum(m["psnr"] for m in metrics_list) / num_views
        scene_ssim = sum(m["ssim"] for m in metrics_list) / num_views
        scene_niqe = float(np.nanmean([m["niqe"] for m in metrics_list]))
        scene_brisque = float(np.nanmean([m["brisque"] for m in metrics_list]))

        # Compute FID across all views (requires at least 2 images)
        scene_fid = float("nan")
        if num_views >= 2:
            scene_fid = self.fid_metric.compute(gt_paths, rendered_paths)
        else:
            print(f"Skipping FID: need at least 2 views, got {num_views}")

        # Log to TensorBoard
        self.tb_logger.log_scalar("Metrics/Scene/LPIPS_mean", scene_lpips, step)
        self.tb_logger.log_scalar(
            "Metrics/Scene/CLIP-Distance_mean", scene_clip_d, step
        )
        self.tb_logger.log_scalar("Metrics/Scene/PSNR_mean", scene_psnr, step)
        self.tb_logger.log_scalar("Metrics/Scene/SSIM_mean", scene_ssim, step)
        self.tb_logger.log_scalar("Metrics/Scene/NIQE_mean", scene_niqe, step)
        self.tb_logger.log_scalar("Metrics/Scene/BRISQUE_mean", scene_brisque, step)
        if not np.isnan(scene_fid):
            self.tb_logger.log_scalar("Metrics/Scene/FID", scene_fid, step)

        # Print to console
        print(f"Scene average LPIPS: {scene_lpips}")
        print(f"Scene average CLIP-Distance: {scene_clip_d}")
        print(f"Scene average PSNR: {scene_psnr}")
        print(f"Scene average SSIM: {scene_ssim}")
        print(f"Scene average NIQE: {scene_niqe}")
        print(f"Scene average BRISQUE: {scene_brisque}")
        print(f"Scene FID: {scene_fid}")

        return {
            "lpips_mean": scene_lpips,
            "clip_distance_mean": scene_clip_d,
            "psnr_mean": scene_psnr,
            "ssim_mean": scene_ssim,
            "niqe_mean": scene_niqe,
            "brisque_mean": scene_brisque,
            "fid": scene_fid,
        }

    def log_image_comparison(
        self,
        tag: str,
        gt_path: str,
        rendered_path: str,
        step: int,
        log_images: bool = True,
    ) -> None:
        """
        Log image comparison to TensorBoard if enabled.

        Args:
            tag: TensorBoard tag for the image
            gt_path: Path to ground truth image
            rendered_path: Path to rendered image
            step: Global step for TensorBoard logging
            log_images: Whether to log images (default: True)
        """
        if log_images:
            self.tb_logger.log_image_comparison(tag, gt_path, rendered_path, step)
