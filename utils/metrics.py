# Code for PSNR and SSIM taken from
# https://github.com/jsh-me/psnr-ssim-tool/tree/master

import cv2
import numpy as np
from skimage.metrics import structural_similarity


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
    print("PSNR: ", psnr)
    print("SSIM: ", ssim)
    return psnr, ssim
