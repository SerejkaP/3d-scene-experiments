import os
import torch
from worldgen import WorldGen
from PIL import Image


def worldgen_generate(pano_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    low_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) < 24
    worldgen = WorldGen(
        mode="t2s",
        use_sharp=False,
        inpaint_bg=False,
        resolution=1600,
        device=device,
        low_vram=low_vram,
    )
    pano_image = Image.open(pano_path).convert("RGB")
    pano_image = pano_image.resize((2048, 1024))
    splat = worldgen._generate_world(pano_image=pano_image)
    print(f"Saving rendered GS {save_path}")
    splat.save(save_path)
    print("Saving successfully!!!")
