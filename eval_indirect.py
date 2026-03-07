import json
import os
from gaussian_renderer import render_radiogs
import numpy as np
import torch
import torch.nn.functional as F
from scene import GaussianModel, RadioGSModel
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
from os import makedirs
import torchvision
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import csv


def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


def get_mae(gt_normal, normal):
    mae = (gt_normal*normal).sum(0).clamp(-1, 1).arccos().mean() * 180 / np.pi
    return mae
    
if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--albedo_rescale", default=2, type=int, help="0: no scale; 1: single channel scale; 2: three channel scale")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--no_save", default=False, action='store_true')
    parser.add_argument("--no_lpips", default=False, action='store_true')
    parser.add_argument("--data_type", default='tensoir', type=str, choices=['nerf_syn', 'tensoir'])
    parser.add_argument("--parse_only", default=False, action='store_true')
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    # load gaussians
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
    
    psnr_indirect = 0.0
    ssim_indirect = 0.0
    lpips_indirect = 0.0
    results_dict = {}
    normal_bg = torch.tensor([0.0, 0.0, 1.0], device='cuda')

    if not args.parse_only:    
        # <<< CHANGE START >>>
        render_times = []
        # <<< CHANGE END >>>

        pbar = tqdm(frames, desc="Evaluating", dynamic_ncols=True)

        for idx, frame in enumerate(pbar):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            
            gt_path = os.path.join(args.source_path, frame["file_path"]+".png")
            gt_rgb_np = load_img_rgb(gt_path)
            mask = torch.from_numpy(gt_rgb_np[..., 3:]).permute(2, 0, 1).float().cuda()

            if args.data_type == 'tensoir':
                gt_indirect_path = os.path.join(args.source_path, frame["file_path"].replace("rgba", "indirect.png"))
            elif args.data_type == 'nerf_syn':
                # add "_indirect_0000" before ".png"
                base, ext = os.path.splitext(frame["file_path"])
                gt_indirect_path = os.path.join(args.source_path, base + "_indirect_0000" + ext+".png")
            gt_indirect_np = load_img_rgb(gt_indirect_path)
            gt_indirect = torch.from_numpy(gt_indirect_np[..., :3]).permute(2, 0, 1).float().cuda()

            indirect_path = os.path.join(args.model_path, "test", f"ours_{loaded_iter:05d}", "light_indirect", f"{idx:05d}.png")
            indirect_np = load_img_rgb(indirect_path)
            indirect = torch.from_numpy(indirect_np[..., :3]).permute(2, 0, 1).float().cuda()

            gt_indirect = gt_indirect * mask
            indirect = indirect * mask
            
            H = gt_indirect.shape[1]
            W = gt_indirect.shape[2]
            fovy = focal2fov(fov2focal(fovx, W), H)

            custom_cam = Camera(colmap_id=0, R=R, T=T,
                                FoVx=fovx, FoVy=fovy,
                                image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name=None, uid=0)

            psnr_indirect += psnr(indirect, gt_indirect).mean().double().item()
            ssim_indirect += ssim(indirect, gt_indirect).mean().double().item()
            
            # <<< CHANGE START >>>
            postfix_dict = {
                'PSNR': f'{psnr_indirect / (idx + 1):.2f}',
                'SSIM': f'{ssim_indirect / (idx + 1):.3f}'
            }

            if not args.no_lpips:
                lpips_indirect += lpips(indirect, gt_indirect, net_type='vgg').mean().double().item()
                postfix_dict['LPIPS'] = f'{lpips_indirect / (idx + 1):.3f}'
            
            pbar.set_postfix(postfix_dict)
            # <<< CHANGE END >>>

        psnr_indirect /= len(frames)
        ssim_indirect /= len(frames)
        lpips_indirect /= len(frames)

        # <<< CHANGE START >>>
        avg_render_time_ms = np.mean(render_times)
        total_render_time_ms = np.sum(render_times)
        fps = 1000.0 / avg_render_time_ms
        # <<< CHANGE END >>>
        results_dict["psnr_indirect_avg"] = psnr_indirect
        results_dict["ssim_indirect_avg"] = ssim_indirect
        results_dict["lpips_indirect_avg"] = lpips_indirect

        
        print("\nEvaluating AVG: PSNR_indirect {: .2f} SSIM_indirect {: .3f} LPIPS_indirect {: .3f}".format(psnr_indirect, ssim_indirect, lpips_indirect))
        # <<< CHANGE START >>>
        
        with open(os.path.join(args.model_path, "indirect_results.json"), "w") as f:
            json.dump(results_dict, f, indent=4)
        print("Results saved to", os.path.join(args.model_path, "indirect_results.json"))

    indirect_path = os.path.join(args.model_path, "indirect_results.json")
    nvs_path = os.path.join(args.model_path, "test", "nvs_results.json")
    out_csv = os.path.join(args.model_path, "results.csv")

    # --- Load indirect ---
    if not os.path.exists(indirect_path):
        raise FileNotFoundError(f"Not found: {indirect_path}")
    with open(indirect_path, "r") as f:
        indirect = json.load(f)
    
    # --- Load NVS ---
    if not os.path.exists(nvs_path):
        raise FileNotFoundError(f"Not found: {nvs_path}")
    with open(nvs_path, "r") as f:
        nvs = json.load(f)

    # --- Extract required fields ---
    row = {
        "psnr_avg": nvs.get("psnr_avg"),
        "ssim_avg": nvs.get("ssim_avg"),
        "lpips_avg": nvs.get("lpips_avg"),

        "psnr_indirect_avg": indirect.get("psnr_indirect_avg"),
        "ssim_indirect_avg": indirect.get("ssim_indirect_avg"),
        "lpips_indirect_avg": indirect.get("lpips_indirect_avg")
    }

    # --- Write CSV ---
    header = list(row.keys())
    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerow(row)

    print(f"[OK] CSV saved at: {out_csv}")