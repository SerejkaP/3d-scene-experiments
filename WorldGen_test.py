import os
import torch
from worldgen import WorldGen
from PIL import Image

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

pano_image = Image.open(
    "/mnt/d/datasets/2D-3D-Semantics/area_1/pano/rgb/camera_00d10d86db1e435081a837ced388375f_office_24_frame_equirectangular_domain_rgb.png"
).convert("RGB")
pano_image = pano_image.resize((2048, 1024))
splat = worldgen._generate_world(pano_image=pano_image)

save_path = os.path.join("/mnt/e/3D/experiments/output/WorldGen", "splat.ply")
splat.save(save_path)
