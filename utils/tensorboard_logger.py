import torch
import cv2
from torch.utils.tensorboard import SummaryWriter
from typing import Optional


class TensorBoardLogger:
    def __init__(
        self,
        log_dir: Optional[str],
        enabled: bool,
        flush_secs: int = 30,
        max_image_size: int = 2048,
    ):
        self.writer = None
        self.enabled = enabled
        if self.enabled:
            self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self.max_image_size = max_image_size

    def log_scalar(self, tag: str, value: float, step: int):
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def _load_image_as_tensor(self, image_path: str) -> torch.Tensor:
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # downscale
        h, w = img.shape[:2]
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # HWC -> CHW, [0, 255] -> [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_tensor

    def log_image(self, tag: str, image_path: str, step: int):
        if self.enabled and self.writer:
            img_tensor = self._load_image_as_tensor(image_path)
            self.writer.add_image(tag, img_tensor, step)

    def log_image_comparison(
        self, tag_prefix: str, gt_path: str, rendered_path: str, step: int
    ):
        if self.enabled and self.writer:
            gt_tensor = self._load_image_as_tensor(gt_path)
            rendered_tensor = self._load_image_as_tensor(rendered_path)

            self.writer.add_image(f"{tag_prefix}/ground_truth", gt_tensor, step)
            self.writer.add_image(f"{tag_prefix}/rendered", rendered_tensor, step)

    def close(self):
        if self.enabled and self.writer:
            self.writer.close()
