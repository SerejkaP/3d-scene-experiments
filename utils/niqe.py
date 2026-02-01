"""
Standalone NIQE (Natural Image Quality Evaluator) implementation.

Extracted from BasicSR (https://github.com/xinntao/BasicSR) and adapted
to remove all BasicSR dependencies.

Paper: Making a "Completely Blind" Image Quality Analyzer.
Reference MATLAB: http://live.ece.utexas.edu/research/quality/niqe_release.zip

MATLAB R2021a result for baboon.png: 5.7296
This implementation result for baboon.png:  5.7296
"""

import cv2
import math
import numpy as np
import os
from scipy.ndimage import convolve
from scipy.special import gamma


def _bgr2ycbcr_y(img):
    """Convert BGR image to Y channel of YCbCr (MATLAB-compatible).

    Args:
        img (ndarray): BGR image in [0, 1] float range.

    Returns:
        ndarray: Y channel in [0, 1] float range.
    """
    return np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0 / 255.0


def _reorder_image(img, input_order="HWC"):
    """Reorder images to 'HWC' order."""
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(
            f"Wrong input_order {input_order}. "
            "Supported input_orders are 'HWC' and 'CHW'"
        )
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == "CHW":
        img = img.transpose(1, 2, 0)
    return img


def _to_y_channel(img):
    """Convert image to Y channel of YCbCr.

    Args:
        img (ndarray): Image with range [0, 255].

    Returns:
        ndarray: Y channel image with range [0, 255] (float).
    """
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[2] == 3:
        img = _bgr2ycbcr_y(img)
        img = img[..., None]
    return img * 255.0


def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.

    Args:
        block (ndarray): 2D image block.

    Returns:
        tuple: alpha, beta_l, beta_r for the AGGD distribution.
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (
        gamma(gam_reciprocal) * gamma(gam_reciprocal * 3)
    )

    left_std = np.sqrt(np.mean(block[block < 0] ** 2))
    right_std = np.sqrt(np.mean(block[block > 0] ** 2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block))) ** 2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1) ** 2)
    array_position = np.argmin((r_gam - rhatnorm) ** 2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)


def compute_feature(block):
    """Compute 18-dimensional feature vector from an image block.

    Args:
        block (ndarray): 2D image block.

    Returns:
        list: Features with length of 18.
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(block, shifts[i], axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def niqe_core(
    img,
    mu_pris_param,
    cov_pris_param,
    gaussian_window,
    block_size_h=96,
    block_size_w=96,
):
    """Calculate NIQE metric on a grayscale image.

    Args:
        img (ndarray): Grayscale input image (h, w), range [0, 255], float.
        mu_pris_param (ndarray): Mean of pristine MVG model.
        cov_pris_param (ndarray): Covariance of pristine MVG model.
        gaussian_window (ndarray): 7x7 Gaussian smoothing window.
        block_size_h (int): Block height. Default: 96.
        block_size_w (int): Block width. Default: 96.

    Returns:
        float: NIQE score (lower is better).
    """
    assert (
        img.ndim == 2
    ), "Input image must be a gray or Y (of YCbCr) image with shape (h, w)."
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0 : num_block_h * block_size_h, 0 : num_block_w * block_size_w]

    distparam = []
    for scale in (1, 2):
        mu = convolve(img, gaussian_window, mode="nearest")
        sigma = np.sqrt(
            np.abs(
                convolve(np.square(img), gaussian_window, mode="nearest")
                - np.square(mu)
            )
        )
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                block = img_nomalized[
                    idx_h * block_size_h // scale : (idx_h + 1) * block_size_h // scale,
                    idx_w * block_size_w // scale : (idx_w + 1) * block_size_w // scale,
                ]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))

        if scale == 1:
            # Downscale by 0.5 using INTER_AREA (closest to MATLAB imresize)
            h_s, w_s = img.shape
            img = cv2.resize(img, (w_s // 2, h_s // 2), interpolation=cv2.INTER_AREA)

    distparam = np.concatenate(distparam, axis=1)

    mu_distparam = np.nanmean(distparam, axis=0)
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    # Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param),
        np.transpose((mu_pris_param - mu_distparam)),
    )

    quality = np.sqrt(quality)
    quality = float(np.squeeze(quality))
    return quality


# Load pristine parameters once at module level
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_niqe_pris_params = np.load(os.path.join(_ROOT_DIR, "niqe_pris_params.npz"))
_MU_PRIS_PARAM = _niqe_pris_params["mu_pris_param"]
_COV_PRIS_PARAM = _niqe_pris_params["cov_pris_param"]
_GAUSSIAN_WINDOW = _niqe_pris_params["gaussian_window"]


def calculate_niqe(img, crop_border=0, input_order="HWC", convert_to="y"):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Args:
        img (ndarray): Input image in range [0, 255] (float/int).
            Input order can be 'HW', 'HWC' or 'CHW'. (BGR order for color)
        crop_border (int): Cropped pixels in each edge. Default: 0.
        input_order (str): 'HW', 'HWC' or 'CHW'. Default: 'HWC'.
        convert_to (str): 'y' (MATLAB YCbCr) or 'gray'. Default: 'y'.

    Returns:
        float: NIQE score (lower is better).
    """
    img = img.astype(np.float32)
    if input_order != "HW":
        img = _reorder_image(img, input_order=input_order)
        if convert_to == "y":
            img = _to_y_channel(img)
        elif convert_to == "gray":
            img = cv2.cvtColor(img / 255.0, cv2.COLOR_BGR2GRAY) * 255.0
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    img = img.round()

    return niqe_core(img, _MU_PRIS_PARAM, _COV_PRIS_PARAM, _GAUSSIAN_WINDOW)
