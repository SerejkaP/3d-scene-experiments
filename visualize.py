"""Gaussian splats

Viser includes a WebGL-based Gaussian splat renderer.

**Features:**

* :meth:`viser.SceneApi.add_gaussian_splats` to add a Gaussian splat object
* Correct sorting when multiple splat objects are present
* Compositing with other scene objects

.. note::
    This example requires external assets. To download them, run:

    .. code-block:: bash

        git clone https://github.com/nerfstudio-project/viser.git
        cd viser/examples
        ./assets/download_assets.sh
        python 01_scene/09_gaussian_splats.py  # With viser installed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple, TypedDict

import imageio
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm
import tyro
from plyfile import PlyData

import viser
from viser import transforms as tf
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import trimesh


class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""

    centers: npt.NDArray[np.floating]
    """(N, 3)."""
    rgbs: npt.NDArray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: npt.NDArray[np.floating]
    """(N, 1). Range [0, 1]."""
    covariances: npt.NDArray[np.floating]
    """(N, 3, 3)."""


def load_splat_file(splat_path: Path, center: bool = False) -> SplatFile:
    """Load an antimatter15-style splat file."""
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    print(
        f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return SplatFile(
        centers=centers,
        # Colors should have shape (N, 3).
        rgbs=splat_uint8[:, 24:27] / 255.0,
        opacities=splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        covariances=covariances,
    )


def quaternion_slerp(q1, q2, t):
    """Spherical linear interpolation between quaternions."""
    q1 = np.array(q1)
    q2 = np.array(q2)

    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.sum(q1 * q2)

    if dot < 0.0:
        q2 = -q2
        dot = -dot

    dot = min(1.0, max(-1.0, dot))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    if sin_theta < 1e-6:
        return q1 * (1 - t) + q2 * t

    s1 = np.sin((1 - t) * theta) / sin_theta
    s2 = np.sin(t * theta) / sin_theta

    return q1 * s1 + q2 * s2


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:
    """Load Gaussians stored in a PLY file."""
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return SplatFile(
        centers=positions,
        rgbs=colors,
        opacities=opacities,
        covariances=covariances,
    )


class ViserVisualizer:
    def __init__(self):
        self.server = viser.ViserServer()
        self.server.scene.set_up_direction("-y")
        self.server.scene.enable_default_lights(False)
        self.start_camera = None

    def add_camera_frustum(
        self,
        name: str,
        fov: float,
        aspect: float,
        scale: float = 0.2,
        position: Tuple[float, float, float] = (0, 0, 0),
        wxyz: Tuple[float, float, float, float] = (1, 0, 0, 0),
        color: Tuple[int, int, int] = (0, 255, 0),
        visible: bool = True,
    ):
        return self.server.scene.add_camera_frustum(
            name,
            fov=fov,
            aspect=aspect,
            scale=scale,
            position=position,
            wxyz=wxyz,
            color=color,
            visible=visible,
        )

    def add_gs(self, splat_data: SplatFile):
        self.scene_gs_handle = self.server.scene.add_gaussian_splats(
            "/scene_gs",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
        )

    def add_original_camera(self):
        h, w = 1080, 1920
        fov = np.deg2rad(90)
        aspect = w / h
        self.original_camera = self.server.scene.add_camera_frustum(
            "original_camera", fov, aspect
        )
        self.init_h, self.init_w = h, w
        self.original_camera.visible = False

    def prepare_render_visibility(self):
        self.original_camera.visible = False
        for frame in self.frames:
            frame.visible = False

        if hasattr(self, "gs_transform_controls"):
            self.gs_transform_controls.scale = 0.0

    def restore_render_visibility(self):
        self.original_camera.visible = True
        for frame in self.frames:
            frame.visible = True
        if hasattr(self, "gs_transform_controls"):
            self.gs_transform_controls.scale = 2.0

    def save_novel_views(self, client):
        # Get render parameters from UI
        render_h = self.render_height_input.value
        render_w = self.render_width_input.value
        render_fov_deg = self.render_fov_input.value
        render_fov_rad = np.deg2rad(render_fov_deg)

        print(
            f"Starting to save novel views ({render_h}x{render_w}, FoV: {render_fov_deg}°)"
        )
        image_dir = os.path.join(self.args.output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Render all cameras
        start_idx = 0
        self.prepare_render_visibility()
        rgb_video_writer = imageio.get_writer(
            os.path.join(self.args.output_dir, "rgb.mp4"), fps=30
        )
        for i, frame in tqdm(enumerate(self.frames), total=len(self.frames)):
            # Use values from UI for rendering
            rendered_image = client.get_render(
                height=render_h,
                width=render_w,
                wxyz=frame.wxyz,
                position=frame.position,
                fov=render_fov_rad,  # Use FoV from UI (in radians)
            )
            imageio.imwrite(f"{image_dir}/{i+start_idx:04d}.png", rendered_image)
            rgb_video_writer.append_data(rendered_image)
        rgb_video_writer.close()
        self.restore_render_visibility()

    def add_interpolated_cameras(self, client):
        current_wxyz = client.camera.wxyz
        current_position = client.camera.position

        steps = self.interpolation_steps.value
        current_fov = self.original_camera.fov
        current_aspect = self.original_camera.aspect

        # Add click handler to teleport to this camera view
        def create_click_handler(f):
            def click_handler(_):
                with client.atomic():
                    client.camera.wxyz = f.wxyz
                    client.camera.position = f.position
                    client.camera.fov = f.fov

            return click_handler

        if self.start_camera is None:
            self.start_camera = self.add_camera_frustum(
                "/start_camera",
                fov=current_fov,
                aspect=current_aspect,
                position=current_position,
                wxyz=current_wxyz,
                color=(0, 0, 0),
            )
            self.start_camera.on_click(create_click_handler(self.start_camera))
            self.frames.append(self.start_camera)
            return

        start_wxyz = self.start_camera.wxyz
        start_position = self.start_camera.position
        camera_counter = [0]

        # Create interpolated cameras
        for i in range(1, steps + 1):
            # Calculate interpolation factor (0 to 1)
            t = i / steps

            # Apply ease-in-out effect (cubic bezier)
            t_eased = t * t * (3 - 2 * t)  # Smooth step function for ease-in-out

            interp_position = (
                t_eased * current_position + (1 - t_eased) * start_position
            )
            interp_wxyz = quaternion_slerp(start_wxyz, current_wxyz, t_eased)

            c2w = torch.eye(4, dtype=torch.float64, device=self.device)
            c2w[:3, :3] = torch.from_numpy(
                R.from_quat(interp_wxyz, scalar_first=True).as_matrix()
            )
            c2w[:3, 3] = torch.tensor(interp_position, device=self.device)

            # Create a new camera frustum at the interpolated position
            camera_name = f"/{camera_counter[0]}"
            camera_counter[0] += 1

            # Color gradient from blue to green
            r = int(0 * (1 - t) + 0 * t)
            g = int(150 * (1 - t) + 255 * t)
            b = int(255 * (1 - t) + 0 * t)

            # Add camera frustum
            frustum = self.server.scene.add_camera_frustum(
                camera_name,
                fov=current_fov,
                aspect=current_aspect,
                scale=0.2,
                wxyz=interp_wxyz,
                position=interp_position,
                color=(r, g, b),  # Color gradient from blue to green
            )

            frustum.on_click(create_click_handler(frustum))
            self.frames.append(frustum)

        print(f"Added camera path with {steps+1} cameras")

    def create_ui(self, client):
        initial_fov_rad = self.original_camera.fov
        initial_fov_deg = np.rad2deg(initial_fov_rad)
        client.camera.position = (0, 0, 0)
        client.camera.wxyz = (1, 0, 0, 0)
        client.camera.fov = initial_fov_rad
        client.camera.far = 10000
        client.camera.near = 0.01

        with client.gui.add_folder("Camera Path"):
            self.interpolation_steps = client.gui.add_slider(
                "Interpolation Steps", min=1, max=1000, step=1, initial_value=120
            )
            self.add_camera_path_button = client.gui.add_button("Generate Camera Path")

        with client.gui.add_folder("Render Settings"):
            self.render_fov_input = client.gui.add_number(
                "Render FoV (deg)",
                initial_value=initial_fov_deg,
                min=1.0,
                max=179.0,
                step=5,
            )
            self.render_height_input = client.gui.add_number(
                "Render Height", initial_value=self.init_h, min=64, max=4096, step=1
            )
            self.render_width_input = client.gui.add_number(
                "Render Width", initial_value=self.init_w, min=64, max=4096, step=1
            )
            self.save_button = client.gui.add_button("Save Novel Views")

            # Update client camera FoV when the input changes
            @self.render_fov_input.on_update
            def _(value):
                client.camera.fov = np.deg2rad(self.render_fov_input.value)

    def set_bg(self, splat: SplatFile):
        # Use black background
        bg_img = np.zeros((1, 1, 3))
        self.server.scene.set_background_image(bg_img)

    def run(self, splat_path: Path):

        if splat_path.suffix == ".splat":
            splat_data = load_splat_file(splat_path, center=True)
        elif splat_path.suffix == ".ply":
            splat_data = load_ply_file(splat_path, center=True)
        else:
            raise SystemExit("Please provide a filepath to a .splat or .ply file.")

        self.add_gs(splat_data)
        self.set_bg(splat_data)
        self.add_original_camera()

        print("\033[92m" + "=" * 70 + "\033[0m")
        print(
            "\033[95mOpen your browser at http://localhost:8080 to view the scene\033[0m"
        )
        print("\033[92m" + "=" * 70 + "\033[0m")

        @self.server.on_client_connect
        def connect(client: viser.ClientHandle) -> None:
            self.create_ui(client)

            @self.original_camera.on_click
            def _(_):
                with client.atomic():
                    client.camera.wxyz = self.original_camera.wxyz
                    client.camera.position = self.original_camera.position
                    client.camera.fov = self.original_camera.fov

            @self.save_button.on_click
            def _(_):
                self.save_novel_views(client)

            @self.add_camera_path_button.on_click
            def _(_):
                self.add_interpolated_cameras(client)

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Exiting...")


def main(splat_path: Path) -> None:
    server = ViserVisualizer()
    server.run(splat_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Viser visualization for gsplats")
    parser.add_argument("--scene", "-s", type=str, help="Path for input scene")
    args = parser.parse_args()
    scene = Path(args.scene)
    main(scene)
