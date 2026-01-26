import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image


def compute_gt_metrics(gt_image_path, pred_image_path):
    gt_image = np.array(Image.open(gt_image_path))
    rendered_image = np.array(Image.open(pred_image_path))
    # PSNR
    psnr = peak_signal_noise_ratio(gt_image, rendered_image, data_range=1.0)
    # SSIM
    ssim = structural_similarity(
        gt_image, rendered_image, channel_axis=-1, data_range=1.0
    )
    print("PSNR: ", psnr)
    print("SSIM: ", ssim)
    return psnr, ssim
