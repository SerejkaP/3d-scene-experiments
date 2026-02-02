# Возможно, стоит заменить на pyiqa, но есть проблемы зависимостей
# https://github.com/chaofengc/IQA-PyTorch

from typing import Optional, List
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.io import imread
import lpips
import clip
from PIL import Image
import piq
from piq.feature_extractors import InceptionV3
from .niqe import calculate_niqe


class ClipDistanceMetric:
    def __init__(
        self,
        device: Optional[torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)
        self.model.eval()

    def compute(self, gt_image_path: str, render_image_path: str):
        contrast = Image.open(render_image_path).convert("RGB")
        original = Image.open(gt_image_path).convert("RGB")
        img1 = self.preprocess(contrast).unsqueeze(0).to(self.device)
        img2 = self.preprocess(original).unsqueeze(0).to(self.device)

        with torch.no_grad():
            f1 = self.model.encode_image(img1)
            f2 = self.model.encode_image(img2)

        f1 = f1 / f1.norm(dim=-1, keepdim=True)
        f2 = f2 / f2.norm(dim=-1, keepdim=True)

        clip_dist = 1 - (f1 @ f2.T).item()
        return clip_dist


class LpipsMetric:
    def __init__(
        self,
        device: Optional[torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.device = device
        self.loss_fn = lpips.LPIPS(net="alex").to(self.device)
        self.loss_fn.eval()

    def compute(self, gt_image_path: str, render_image_path: str):
        gt_image = lpips.load_image(gt_image_path)
        render_image = lpips.load_image(render_image_path)
        if gt_image.shape[-1] == 4:
            gt_image = gt_image[..., :3]
        if render_image.shape[-1] == 4:
            render_image = render_image[..., :3]
        gt_height, gt_width = gt_image.shape[:2]
        render_height, render_width = render_image.shape[:2]
        if (gt_height, gt_width) != (render_height, render_width):
            render_pil = Image.fromarray((render_image * 255).astype(np.uint8))
            render_pil = render_pil.resize((gt_width, gt_height), Image.LANCZOS)
            render_image = np.array(render_pil).astype(np.float32) / 255.0
        gt_image_tensor = lpips.im2tensor(gt_image).to(self.device)
        render_image_tensor = lpips.im2tensor(render_image).to(self.device)
        with torch.no_grad():
            lpips_score = self.loss_fn.forward(gt_image_tensor, render_image_tensor)

        return lpips_score.cpu().item()


def compute_gt_metrics(gt_image_path: str, pred_image_path: str):
    gt_image = imread(gt_image_path)
    pred_image = imread(pred_image_path)

    # RGBA -> RGB
    if gt_image.shape[-1] == 4:
        gt_image = gt_image[..., :3]
    if pred_image.shape[-1] == 4:
        pred_image = pred_image[..., :3]

    if gt_image.shape[:2] != pred_image.shape[:2]:
        pred_image = cv2.resize(
            pred_image,
            dsize=(gt_image.shape[1], gt_image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    gt_tensor = (
        torch.tensor(gt_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        / 255.0
    )
    pred_tensor = (
        torch.tensor(pred_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        / 255.0
    )
    psnr_score = piq.psnr(pred_tensor, gt_tensor, data_range=1.0, reduction="none")
    ssim_score = piq.ssim(pred_tensor, gt_tensor, data_range=1.0)

    return psnr_score.item(), ssim_score.item()


class _ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return {"images": self.transform(img)}


class FidMetric:
    def __init__(
        self,
        device: Optional[torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.device = device
        self.fid = piq.FID()
        self.feature_extractor = InceptionV3(use_fid_inception=True).to(self.device)
        self.feature_extractor.eval()

    def compute(
        self,
        gt_image_paths: List[str],
        render_image_paths: List[str],
        batch_size: int = 16,
    ):
        gt_dataset = _ImagePathDataset(gt_image_paths)
        render_dataset = _ImagePathDataset(render_image_paths)

        gt_loader = DataLoader(gt_dataset, batch_size=batch_size, num_workers=0)
        render_loader = DataLoader(render_dataset, batch_size=batch_size, num_workers=0)

        with torch.no_grad():
            gt_feats = self.fid.compute_feats(
                gt_loader,
                feature_extractor=self.feature_extractor,
                device=self.device,
            )
            render_feats = self.fid.compute_feats(
                render_loader,
                feature_extractor=self.feature_extractor,
                device=self.device,
            )

        return self.fid(gt_feats, render_feats).item()


def compute_niqe(image_path: str, crop_border: int = 0):
    image = cv2.imread(image_path)
    if image.shape[-1] == 4:
        image = image[..., :3]
    try:
        return calculate_niqe(
            image, crop_border=crop_border, input_order="HWC", convert_to="y"
        )
    except np.linalg.LinAlgError:
        return float("nan")


def compute_brisque(image_path: str):
    image = imread(image_path)
    if image.shape[-1] == 4:
        image = image[..., :3]
    tensor = (
        torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    )
    with torch.no_grad():
        score = piq.brisque(tensor, data_range=1.0, reduction="none")
    return score.item()


class DepthMetrics:
    def __init__(self, min_depth: float = 0.0, max_depth: float = 100.0):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def compute(
        self,
        gt_depth: np.ndarray,
        pred_depth: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ):
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.squeeze(-1)
        if pred_depth.ndim == 3:
            pred_depth = pred_depth.squeeze(-1)
        if gt_depth.shape != pred_depth.shape:
            pred_depth = cv2.resize(
                pred_depth,
                (gt_depth.shape[1], gt_depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        if mask is None:
            mask = np.ones_like(gt_depth, dtype=bool)
        else:
            mask = mask.astype(bool)
        valid_mask = (
            mask
            & (gt_depth > self.min_depth)
            & (gt_depth < self.max_depth)
            & (pred_depth > self.min_depth)
            & (pred_depth < self.max_depth)
        )

        if not valid_mask.any():
            return {
                "rmse": float("nan"),
                "abs_rel": float("nan"),
                "sq_rel": float("nan"),
                "rmse_log": float("nan"),
            }

        gt_valid = gt_depth[valid_mask]
        pred_valid = pred_depth[valid_mask]

        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
        # Abs Rel (Absolute Relative Error)
        abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
        # Sq Rel (Squared Relative Error)
        sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
        # RMSE log
        rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))

        return {
            "rmse": float(rmse),
            "abs_rel": float(abs_rel),
            "sq_rel": float(sq_rel),
            "rmse_log": float(rmse_log),
        }
