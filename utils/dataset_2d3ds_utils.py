import os

# camera_00d10d86db1e435081a837ced388375f_office_24_frame_equirectangular_domain_rgb.png
# camera_00d10d86db1e435081a837ced388375f_office_24_frame_equirectangular_domain_pose.json

# camera_00d10d86db1e435081a837ced388375f_office_24_frame_0_domain_pose.json
# camera_00d10d86db1e435081a837ced388375f_office_24_frame_0_domain_rgb.png


def pose_json_by_image_path(pano_rgb_name: str, pose_dir_path):
    pano_name_pieces = pano_rgb_name.split("_")
    pano_name = "_".join(pano_name_pieces[:5])
    pano_json_path = os.path.join(
        pose_dir_path, f"{pano_name}_equirectangular_domain_pose.json"
    )
    return pano_json_path, pano_name