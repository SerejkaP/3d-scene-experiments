import wandb
import cv2
import numpy as np
from typing import Optional


class WandbLogger:
    def __init__(
        self,
        log_dir: Optional[str],
        enabled: bool,
        project: str = "3d-generation",
        max_image_size: int = 2048,
    ):
        self.enabled = enabled
        self.max_image_size = max_image_size
        self.run = None

        if self.enabled:
            self.run = wandb.init(
                project=project,
                dir=log_dir,
                mode="offline",
            )

    def log_scalar(self, tag: str, value: float, step: int):
        if self.enabled and self.run:
            wandb.log({tag: value})

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        if self.enabled and self.run:
            log_dict = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
            wandb.log(log_dict)

    def _load_and_resize_image(self, image_path: str) -> np.ndarray:
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    def log_image(self, tag: str, image_path: str, step: int):
        if self.enabled and self.run:
            img = self._load_and_resize_image(image_path)
            wandb.log({tag: wandb.Image(img)})

    def log_image_comparison(
        self, tag_prefix: str, gt_path: str, rendered_path: str, step: int
    ):
        if self.enabled and self.run:
            gt_img = self._load_and_resize_image(gt_path)
            rendered_img = self._load_and_resize_image(rendered_path)
            wandb.log(
                {
                    f"{tag_prefix}/ground_truth": wandb.Image(gt_img),
                    f"{tag_prefix}/rendered": wandb.Image(rendered_img),
                }
            )

    def close(self):
        if self.enabled and self.run:
            wandb.finish()
            self.run = None
