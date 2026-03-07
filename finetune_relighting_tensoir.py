import json
import sys
import os
from gaussian_renderer import render_radiogs
import numpy as np
import torch
from gaussian_renderer.radiogs import sample_incident_rays
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
from utils.system_utils import Timing
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
    parser.add_argument('-bg', "--background_color", type=float, default=1,
                        help="If set, use it as background color")
    parser.add_argument("--albedo_rescale", default=2, type=int, help="0: no scale; 1: single channel scale; 2: three channel scale")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--no_save", default=False, action='store_true')
    parser.add_argument("--no_lpips", default=False, action='store_true')
    parser.add_argument("-e", "--extra", default='', type=str)
    parser.add_argument("--finetune", default=False, action='store_true')
    parser.add_argument("--envmaps", default=[], nargs='+', type=str, help="If provided, only use the specified envmaps for relighting (without path and extension)")
    parser.add_argument("--start", default=0, type=int)
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
        
    # deal with each item
    test_transforms_file = os.path.join(args.source_path, "transforms_test.json")
    contents = load_json_config(test_transforms_file)

    fovx = contents["camera_angle_x"]
    frames = contents["frames"]
    
    # Define the base path
    envmap_dir = "data/Environment_Maps/high_res_envmaps_2k"

    # List all EXR files in the directory
    if len(args.envmaps) > 0:
        hdr_files = [f + '.hdr' for f in args.envmaps]
    else:
        hdr_files = [f+'.hdr' for f in ["bridge", "city", "fireplace", "forest", "night"]]
    print(hdr_files)

    # Create the task dictionary dynamically
    task_dict = {
        f"envmap_{hdr_file[:-4]}": {
            "capture_list": ["render", "render_sh", "render_env", "render_direct", "render_indirect", "light_direct", "light_indirect"],
            "envmap_path": os.path.join(envmap_dir, hdr_file),
        }
        for hdr_file in hdr_files
    }
    results_dict = {}

    bg = 1 if dataset.white_background else 0
    background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    
    results_dir = os.path.join(args.model_path, "test_rli" + (f"_{args.extra}" if len(args.extra)>0 else ""))
    
    if args.albedo_rescale == 0:
        base_color_scale = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    else:
        with open(os.path.join(args.model_path, "albedo_scale.json"), "r") as f:
            albedo_scale_dict = json.load(f)
        base_color_scale = torch.tensor(albedo_scale_dict[str(args.albedo_rescale)], dtype=torch.float32, device="cuda")
    
    for task_name in task_dict:

        # model_path = os.path.join(args.model_path, "finetune", task_name)
        model_path = os.path.join(args.model_path, args.extra, task_name)
        # load gaussians
        # 
        gaussians = RadioGSModel(3)
        

        if args.iteration < 0:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        else:
            loaded_iter = args.iteration
        gaussians.load_ply(os.path.join(model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply")) 
        gaussians.build_bvh()

        results_dict[task_name] = {}
        task_dir = os.path.join(results_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        gaussians.env_map = EnvLight(path=task_dict[task_name]["envmap_path"], device='cuda', max_res=1024, activation='none').cuda()
        gaussians.env_map.build_mips()
        gaussians.env_map.update_pdf()
        transform = torch.tensor([
            [0, -1, 0], 
            [0, 0, 1], 
            [-1, 0, 0]
        ], dtype=torch.float32, device="cuda")
        gaussians.env_map.set_transform(transform)

        render_kwargs = {
            "pc": gaussians,
            "pipe": pipe,
            "bg_color": background,
            "training": False,
            "relight": True,
            "base_color_scale": base_color_scale,
        }
        
        psnr_pbr = 0.0
        ssim_pbr = 0.0
        lpips_pbr = 0.0
        psnr_sh = 0.0
        ssim_sh = 0.0
        lpips_sh = 0.0
        
        capture_list = task_dict[task_name]["capture_list"]
        if not args.no_save:
            for capture_type in capture_list:
                capture_type_dir = os.path.join(task_dir, capture_type)
                os.makedirs(capture_type_dir, exist_ok=True)
            os.makedirs(os.path.join(task_dir, "render_white"), exist_ok=True)
            os.makedirs(os.path.join(task_dir, "render_sh_white"), exist_ok=True)

            os.makedirs(os.path.join(task_dir, "gt"), exist_ok=True)
            os.makedirs(os.path.join(task_dir, "gt_env"), exist_ok=True)
            
        envname = os.path.splitext(os.path.basename(task_dict[task_name]["envmap_path"]))[0]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            if idx < args.start: continue
            image_path = os.path.join(args.source_path, f"test_{idx:03}/" + frame["file_path"].split("/")[-1] + "_" + envname + ".png")
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_rgba = load_img_rgb(image_path)
            image = image_rgba[..., :3]
            mask = image_rgba[..., 3:]
            gt_image = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float().cuda()
            gt_image = gt_image * mask + bg * (1 - mask)
            
            H = image.shape[0]
            W = image.shape[1]
            fovy = focal2fov(fov2focal(fovx, W), H)

            custom_cam = Camera(colmap_id=0, R=R, T=T,
                                FoVx=fovx, FoVy=fovy,
                                image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name=None, uid=0)
            
            # precompute indirect light for relighting with split-sum approach from IRGS
            # note that direct rasterization from Gaussian SH yiels finetuned relighting result
            if idx == args.start and True:
                with torch.no_grad():
                    dir_pp = gaussians.get_xyz - custom_cam.camera_center
                    dir_pp_normalized = dir_pp / (torch.norm(dir_pp, dim=-1, keepdim=True) + 1e-6)
                    B = dir_pp_normalized.shape[0]
                    normal_dummy = gaussians.get_normal(scaling_modifier=1.0, dir_pp_normalized=dir_pp_normalized)
                    p_diffuse = pipe.diffuse_sample_num / (pipe.diffuse_sample_num + pipe.light_sample_num)
                    p_light = pipe.light_sample_num / (pipe.diffuse_sample_num + pipe.light_sample_num)
                    diffuse_directions, diffuse_areas = sample_incident_rays(normal_dummy, False, pipe.diffuse_sample_num)
                    diffuse_pdfs = 1 / diffuse_areas
                    light_directions, light_pdfs = gaussians.get_envmap.sample_light_directions(B, pipe.light_sample_num, False)
                    diffuse_pdfs_light = 1 / (2 * np.pi)
                    light_pdfs_diffuse = gaussians.get_envmap.light_pdf(diffuse_directions)
                    diffuse_pdfs = diffuse_pdfs * p_diffuse + light_pdfs_diffuse * p_light
                    light_pdfs = diffuse_pdfs_light * p_diffuse + light_pdfs * p_light
                    incident_dirs = torch.cat([diffuse_directions, light_directions], dim=1)
                    incident_pdfs = torch.cat([diffuse_pdfs, light_pdfs], dim=1)
                    incident_areas = 1 / incident_pdfs.clamp_min(1e-6)
                    gaussians.update_incidents_directions(incident_dirs, incident_areas)
                    features = torch.cat([gaussians.get_base_color, gaussians.get_rough], dim=1)
                    gaussians.precompute_incidents(light_t_min=pipe.light_t_min, only_vis=False, features=features, relight=(not args.finetune), back_culling=pipe.back_culling)

            with torch.no_grad():
                render_pkg = render_radiogs(viewpoint_camera=custom_cam, **render_kwargs)

            render_pkg["render"] = render_pkg["render"] * mask + (1 - mask) * bg
            if "render_sh" in render_pkg:
                render_pkg["render_sh"] = render_pkg["render_sh"] * mask + (1 - mask) * bg
            gt_image_env = gt_image + render_pkg["env_only"] * (1 - mask)
            if not args.no_save:
                save_image(gt_image, os.path.join(task_dir, "gt", f"{idx}.png"))
                save_image(gt_image_env, os.path.join(task_dir, "gt_env", f"{idx}.png"))
                for capture_type in capture_list:
                    save_image(render_pkg[capture_type], os.path.join(task_dir, capture_type, f"{idx}.png"))
                render_white = render_pkg["render"] + (1 - render_pkg["rend_alpha"])[:, None, :, :]
                save_image(render_white, os.path.join(task_dir, "render_white", f"{idx}.png"))
                if "render_sh" in render_pkg:
                    render_sh_white = render_pkg["render_sh"] + (1 - render_pkg["rend_alpha"])[:, None, :, :]
                    save_image(render_sh_white, os.path.join(task_dir, "render_sh_white", f"{idx}.png"))
            
            with torch.no_grad():
                psnr_pbr += psnr(render_pkg['render'], gt_image).mean().double().item()
                ssim_pbr += ssim(render_pkg['render'], gt_image).mean().double().item()
                if not args.no_lpips:
                    lpips_pbr += lpips(render_pkg['render'], gt_image, net_type='vgg').mean().double().item()
                else:
                    lpips_pbr += 0.0

            if "render_sh" in render_pkg:
                with torch.no_grad():
                    psnr_sh += psnr(render_pkg['render_sh'], gt_image).mean().double().item()
                    ssim_sh += ssim(render_pkg['render_sh'], gt_image).mean().double().item()
                    if not args.no_lpips:
                        lpips_sh += lpips(render_pkg['render_sh'], gt_image, net_type='vgg').mean().double().item()
                    else:
                        lpips_sh += 0.0

            torch.cuda.empty_cache()

            # tqdm.write(f"AVG PBR PSNR: {psnr_pbr / (idx + 1): .4f}")
        psnr_pbr /= len(frames)
        ssim_pbr /= len(frames)
        lpips_pbr /= len(frames)
    
        results_dict[task_name]["psnr_pbr"] = psnr_pbr
        results_dict[task_name]["ssim_pbr"] = ssim_pbr
        results_dict[task_name]["lpips_pbr"] = lpips_pbr

        print("\nEvaluating {}: PSNR_PBR {: .4f} SSIM_PBR {: .4f} LPIPS_PBR {: .4f}".format(task_name, psnr_pbr, ssim_pbr, lpips_pbr))

        psnr_sh /= len(frames)
        ssim_sh /= len(frames)
        lpips_sh /= len(frames)
        results_dict[task_name]["psnr_sh"] = psnr_sh
        results_dict[task_name]["ssim_sh"] = ssim_sh
        results_dict[task_name]["lpips_sh"] = lpips_sh
        print("\nEvaluating {}: PSNR_SH {: .4f} SSIM_SH {: .4f} LPIPS_SH {: .4f}".format(task_name, psnr_sh, ssim_sh, lpips_sh))
        

    task_names = list(task_dict.keys())
    results_dict["psnr_pbr_avg"] = np.mean([results_dict[task_name]["psnr_pbr"] for task_name in task_names])
    results_dict["ssim_pbr_avg"] = np.mean([results_dict[task_name]["ssim_pbr"] for task_name in task_names])
    results_dict["lpips_pbr_avg"] = np.mean([results_dict[task_name]["lpips_pbr"] for task_name in task_names])
    results_dict["psnr_sh_avg"] = np.mean([results_dict[task_name]["psnr_sh"] for task_name in task_names])
    results_dict["ssim_sh_avg"] = np.mean([results_dict[task_name]["ssim_sh"] for task_name in task_names])
    results_dict["lpips_sh_avg"] = np.mean([results_dict[task_name]["lpips_sh"] for task_name in task_names])
    print("\nEvaluating AVG: PSNR_PBR {: .4f} SSIM_PBR {: .4f} LPIPS_PBR {: .4f}".format(results_dict["psnr_pbr_avg"], results_dict["ssim_pbr_avg"], results_dict["lpips_pbr_avg"]))
    print("\nEvaluating AVG: PSNR_SH {: .4f} SSIM_SH {: .4f} LPIPS_SH {: .4f}".format(results_dict["psnr_sh_avg"], results_dict["ssim_sh_avg"], results_dict["lpips_sh_avg"]))
    with open(os.path.join(results_dir, "relighting_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Results saved to", os.path.join(results_dir, "relighting_results.json"))