import json
import os
from gaussian_renderer import render_radiogs
import numpy as np
import torch
from scene import GaussianModel, RadioGSModel, RefGaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import Camera
from scene.light import EnvMap, EnvLight
from utils.graphics_utils import focal2fov, fov2focal, rgb_to_srgb, srgb_to_rgb
from utils.system_utils import searchForMaxIteration
from torchvision.utils import save_image
from tqdm import tqdm
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
from scene.dataset_readers import load_img_rgb
import warnings
warnings.filterwarnings("ignore")


def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    # load gaussians
    # gaussians = GaussianModel(3)
    # 
    gaussians = RadioGSModel(3)
    
    if args.iteration < 0:
        loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    else:
        loaded_iter = args.iteration
    gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply")) 
    gaussians.build_bvh()
    
    # deal with each item
    test_transforms_file = os.path.join(args.source_path, "transforms_test.json")
    contents = load_json_config(test_transforms_file)

    fovx = contents["camera_angle_x"]
    frames = contents["frames"]

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    render_kwargs = {
        "pc": gaussians,
        "pipe": pipe,
        "bg_color": background,
        "training": False,
        "relight": False,
        "base_color_scale": None,
        "material_only": True,
    }

    # if "/air_baloons/" in args.model_path:
    #     gaussians.base_color_scale = torch.tensor([1.3746, 0.6428, 0.7279], dtype=torch.float32, device="cuda") / 1.5
    # elif "/chair/" in args.model_path:
    #     gaussians.base_color_scale = torch.tensor([1.8865, 1.9675, 1.7410], dtype=torch.float32, device="cuda") / 2
    # elif "/hotdog/" in args.model_path:
    #     gaussians.base_color_scale = torch.tensor([2.6734, 2.0917, 1.2587], dtype=torch.float32, device="cuda") / 2
    # elif "/jugs/" in args.model_path:
    #     # gaussians.base_color_scale = torch.tensor([1.1916, 0.9296, 0.5684], dtype=torch.float32, device="cuda")
    #     gaussians.base_color_scale = torch.tensor([1.0044, 0.9253, 0.7648], dtype=torch.float32, device="cuda") / 2
    # elif "/armadillo/" in args.model_path:
    #     gaussians.base_color_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    # render_kwargs["base_color_scale"] = gaussians.base_color_scale
    
    albedo_list = []
    albedo_list2 = []
    albedo_gt_list = []
    albedo_gt_list2 = []
    for idx, frame in enumerate(tqdm(frames, leave=False)):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        albedo_path = os.path.join(args.source_path, frame["file_path"].replace("rgba", "albedo.png"))
        gt_albedo_np = load_img_rgb(albedo_path)
        mask = torch.from_numpy(gt_albedo_np[..., 3:4]).permute(2, 0, 1).float().cuda()
        gt_albedo = torch.from_numpy(gt_albedo_np[..., :3] * gt_albedo_np[..., 3:4]).permute(2, 0, 1).float().cuda()
        mask = torch.logical_and(mask>0, (gt_albedo>0).all(dim=0, keepdim=True))
        
        H = mask.shape[1]
        W = mask.shape[2]
        fovy = focal2fov(fov2focal(fovx, W), H)

        custom_cam = Camera(colmap_id=0, R=R, T=T,
                            FoVx=fovx, FoVy=fovy,
                            image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name=None, uid=0)

        with torch.no_grad():
            render_pkg = render_radiogs(viewpoint_camera=custom_cam, **render_kwargs)


        albedo_gt_list.append(gt_albedo.permute(1, 2, 0)[mask[0] > 0]) # [N, 3]
        albedo_list.append(render_pkg['base_color_linear'].permute(1, 2, 0)[mask[0] > 0]) # [N, 3]
        
    albedo_gts = torch.cat(albedo_gt_list, dim=0) # [N, 3]
    albedo_ours = torch.cat(albedo_list, dim=0) # [N, 3]
    albedo_scale_json = {}
    albedo_scale_json["0"] = [1.0, 1.0, 1.0]
    albedo_scale_json["1"] = [(albedo_gts/albedo_ours.clamp_min(1e-6))[..., 0].median().item()] * 3
    albedo_scale_json["2"] = (albedo_gts/albedo_ours.clamp_min(1e-6)).median(dim=0).values.tolist()
    albedo_scale_json["3"] = (albedo_gts/albedo_ours.clamp_min(1e-6)).mean(dim=0).tolist()
    albedo_scale_json["4"] = (albedo_gt_list[0]/albedo_list[0].clamp_min(1e-6)).median(dim=0).values.tolist()
    print("Albedo scales:\n", albedo_scale_json)
        
    with open(os.path.join(args.model_path, "albedo_scale.json"), "w") as f:
        json.dump(albedo_scale_json, f)