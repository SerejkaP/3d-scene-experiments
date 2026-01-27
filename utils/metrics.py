# Code for PSNR and SSIM taken from
# https://github.com/jsh-me/psnr-ssim-tool/tree/master

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity
import lpips


def psnr_compute(gt_image, render_image):
    mse = np.mean((gt_image - render_image) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def ssim_compute(gt_image, render_image):
    gt_image_gray = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
    render_image_gray = cv2.cvtColor(render_image, cv2.COLOR_BGR2GRAY)

    score, _ = structural_similarity(gt_image_gray, render_image_gray, full=True)
    return score


class LpipsMetric:
    def __init__(self, device=None):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.loss_fn = lpips.LPIPS(net="alex").to(self.device)

    def compute(self, gt_image_path, render_image_path):
        gt_image = lpips.load_image(gt_image_path)
        gt_image_tensor = lpips.im2tensor(gt_image).to(self.device)
        render_image = lpips.load_image(render_image_path)
        render_image_tensor = lpips.im2tensor(render_image).to(self.device)

        lpips_score = self.loss_fn.forward(gt_image_tensor, render_image_tensor)
        return lpips_score.cpu().item()


def compute_gt_metrics(gt_image_path, pred_image_path):
    original = cv2.imread(gt_image_path)
    contrast = cv2.imread(pred_image_path)

    o_height, o_width, _ = original.shape
    contrast = cv2.resize(
        contrast, dsize=(o_width, o_height), interpolation=cv2.INTER_AREA
    )
    # PSNR
    psnr = psnr_compute(original, contrast)
    # SSIM
    ssim = ssim_compute(original, contrast)
    return psnr, ssim
