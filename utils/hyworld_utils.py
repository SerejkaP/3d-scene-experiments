import os
import torch
import open3d as o3d
from hy3dworld import LayerDecomposition
from hy3dworld import WorldComposer, process_file
from hy3dworld.AngelSlim.gemm_quantization_processor import FluxFp8GeMMProcessor
from hy3dworld.AngelSlim.attention_quantization_processor import FluxFp8AttnProcessor2_0


class HYworldDemo:
    def __init__(self, args, seed=42):
        self.args = args
        target_size = 3840
        kernel_scale = max(1, int(target_size / 1920))

        self.LayerDecomposer = LayerDecomposition(args)

        self.hy3d_world = WorldComposer(
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"),
            resolution=(target_size, target_size // 2),
            seed=seed,
            filter_mask=True,
            kernel_scale=kernel_scale,
        )

        if self.args.fp8_attention:
            self.LayerDecomposer.inpaint_fg_model.transformer.set_attn_processor(FluxFp8AttnProcessor2_0())
            self.LayerDecomposer.inpaint_sky_model.transformer.set_attn_processor(FluxFp8AttnProcessor2_0())
        if self.args.fp8_gemm:
            FluxFp8GeMMProcessor(self.LayerDecomposer.inpaint_fg_model.transformer)
            FluxFp8GeMMProcessor(self.LayerDecomposer.inpaint_sky_model.transformer)
            
    def run(self, image_path, labels_fg1, labels_fg2, classes="outdoor", output_dir='output_hyworld', export_drc=False):
        # foreground layer information
        fg1_infos = [
            {
                "image_path": image_path,
                "output_path": output_dir,
                "labels": labels_fg1,
                "class": classes,
            }
        ]
        fg2_infos = [
            {
                "image_path": os.path.join(output_dir, 'remove_fg1_image.png'),
                "output_path": output_dir,
                "labels": labels_fg2,
                "class": classes,
            }
        ]

        # layer decompose
        self.LayerDecomposer(fg1_infos, layer=0)
        self.LayerDecomposer(fg2_infos, layer=1)
        self.LayerDecomposer(fg2_infos, layer=2)
        separate_pano, fg_bboxes = self.hy3d_world._load_separate_pano_from_dir(
            output_dir, sr=True
        )

        # layer-wise reconstruction
        layered_world_mesh = self.hy3d_world.generate_world(
            separate_pano=separate_pano, fg_bboxes=fg_bboxes, world_type='mesh'
        )

        # save results
        for layer_idx, layer_info in enumerate(layered_world_mesh):
            # export ply
            output_path = os.path.join(
                output_dir, f"mesh_layer{layer_idx}.ply"
            )
            o3d.io.write_triangle_mesh(output_path, layer_info['mesh'])

            # export drc
            if export_drc:
                output_path_drc = os.path.join(
                    output_dir, f"mesh_layer{layer_idx}.drc"
                )
                process_file(output_path, output_path_drc)