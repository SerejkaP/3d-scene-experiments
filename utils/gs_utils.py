import torch
import numpy as np
from plyfile import PlyData

def load_gs_ply(path, device):
    plydata = PlyData.read(path)
    v = plydata["vertex"]

    means = torch.tensor(
        np.stack([v["x"], v["y"], v["z"]], axis=-1), device=device, dtype=torch.float32
    )
    colors_dc = torch.tensor(
        np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1),
        device=device,
        dtype=torch.float32,
    )
    colors = 0.5 + 0.28209479177387814 * colors_dc

    opacity = torch.tensor(v["opacity"], device=device, dtype=torch.float32)
    opacity = torch.sigmoid(opacity)

    scales = torch.tensor(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1),
        device=device,
        dtype=torch.float32,
    )
    scales = torch.exp(scales)

    quats = torch.tensor(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1),
        device=device,
        dtype=torch.float32,
    )
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)

    return means, colors, opacity, scales, quats