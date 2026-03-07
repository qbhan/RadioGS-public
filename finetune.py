#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import calculate_radiosity
from gaussian_renderer import render_radiogs, render_finetune
import sys
from scene import Scene, RadioGSModel
from scene.light import EnvMap, EnvLight
from utils.general_utils import safe_state
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from utils.image_utils import visualize_depth
from utils.graphics_utils import rgb_to_srgb, fibonacci_sphere_sampling
from gaussian_renderer.radiogs import sample_incident_rays
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import json
from torchvision.utils import save_image
from tqdm import tqdm
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
from utils.system_utils import Timing
from scene.dataset_readers import load_img_rgb
from utils.graphics_utils import focal2fov, fov2focal, rgb_to_srgb, srgb_to_rgb

def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, checkpoint_refgs, model_path, envmap_path, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger()

    lr_scale = opt.lr_scale
    opt.position_lr_init *= lr_scale
    opt.opacity_lr *= lr_scale
    opt.scaling_lr *= lr_scale
    opt.rotation_lr *= lr_scale
    opt.base_color_lr *= lr_scale
    opt.metallic_lr *= lr_scale
    opt.roughness_lr *= lr_scale
    opt.envmap_cubemap_lr *= lr_scale


    # render = render_radiogs

    gaussians = RadioGSModel(dataset.sh_degree)
    # set_gaussian_para(gaussians, opt)
    
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
    elif checkpoint_refgs:
        (model_params, _) = torch.load(checkpoint_refgs, weights_only=False)
        gaussians.restore_from_refgs(model_params, opt)
        
    gaussians.build_bvh()

    if args.reset_sh: gaussians.reset_features()
    # print(envmap_path)
    envname = os.path.splitext(os.path.basename(envmap_path))[0]
    gaussians.env_map = EnvLight(path=envmap_path, device='cuda', max_res=1024, activation='none').cuda()
    gaussians.env_map.build_mips()
    gaussians.env_map.update_pdf()
    transform = torch.tensor([
        [0, -1, 0], 
        [0, 0, 1], 
        [-1, 0, 0]
    ], dtype=torch.float32, device="cuda")
    gaussians.env_map.set_transform(transform)


    if args.albedo_rescale == 0:
        base_color_scale = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    else:
        scale_path = '/'.join(checkpoint.split('/')[:-1])
        print(scale_path)
        with open(os.path.join(scale_path, "albedo_scale.json"), "r") as f:
            albedo_scale_dict = json.load(f)
        base_color_scale = torch.tensor(albedo_scale_dict[str(args.albedo_rescale)], dtype=torch.float32, device="cuda")
        # save to current folder
        with open(os.path.join(model_path, "albedo_scale.json"), "w") as f:
            json.dump(albedo_scale_dict, f)
    
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_test = 0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    iteration = first_iter

    # update incident visibility per Gaussian
    viewpoint_dummy = scene.getTrainCameras().copy()
    cam = viewpoint_dummy.pop(randint(0, len(viewpoint_dummy) - 1))
    with torch.no_grad():
        dir_pp = gaussians.get_xyz - cam.camera_center
        dir_pp_normalized = dir_pp / (torch.norm(dir_pp, dim=-1, keepdim=True) + 1e-6)
        normal_dummy = gaussians.get_normal(scaling_modifier=1.0, dir_pp_normalized=dir_pp_normalized)
        incident_directions, incident_areas = sample_incident_rays(normal_dummy, is_training=pipe.radiosity_random_sample, sample_num=(pipe.radiosity_sample_num+pipe.light_sample_num))
        gaussians.update_incidents_directions(incident_directions, incident_areas)
        gaussians.precompute_incidents(light_t_min=pipe.light_t_min, only_vis=True, back_culling=pipe.back_culling)
    # gaussians.precompute_incidents(only_vis=False)
    
    while iteration < opt.iterations + 1:
        iter_start.record()

        if iteration % 100 == 0 and iteration != first_iter:
            with torch.no_grad():
                    gaussians.precompute_incidents(light_t_min=pipe.light_t_min, only_vis=False, back_culling=pipe.back_culling)
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTestCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render_finetune(viewpoint_cam, gaussians, pipe, background, opt=opt, iteration=iteration, training=True, base_color_scale=base_color_scale)
        # render_pkg = render_radiogs(viewpoint_cam, gaussians, pipe, background, opt=opt, iteration=iteration, training=True, base_color_scale=base_color_scale)

        gt_image = viewpoint_cam.original_image.cuda()
        
        total_loss, tb_dict = calculate_radiosity(viewpoint_cam, gaussians, render_pkg, opt, iteration)
       
        loss = tb_dict["loss"]
        total_loss.backward()
            
        iter_end.record()

        with torch.no_grad():

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration % 500 == 0 or iteration == first_iter + 1:
                with torch.no_grad():
                    dir_pp = gaussians.get_xyz - cam.camera_center
                    dir_pp_normalized = dir_pp / (torch.norm(dir_pp, dim=-1, keepdim=True) + 1e-6)
                    B = dir_pp_normalized.shape[0]
                    normal_dummy = gaussians.get_normal(scaling_modifier=1.0, dir_pp_normalized=dir_pp_normalized)
                    p_diffuse = pipe.radiosity_sample_num / (pipe.radiosity_sample_num + pipe.light_sample_num)
                    p_light = pipe.light_sample_num / (pipe.radiosity_sample_num + pipe.light_sample_num)
                    diffuse_directions, diffuse_areas = sample_incident_rays(normal_dummy, False, pipe.radiosity_sample_num)
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
                    gaussians.precompute_incidents(light_t_min=pipe.light_t_min, only_vis=False, back_culling=pipe.back_culling)
                save_training_vis(viewpoint_cam, gaussians, background, render_radiogs, pipe, opt, iteration, base_color_scale=base_color_scale)

            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            
            # image = render_pkg["render"]
            # ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
            
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    # "Distort": f"{ema_dist_for_log:.{5}f}",
                    # "Points": f"{gaussians.get_xyz.shape[0]}",
                    # "PSNR-test": f"{psnr_test:.{4}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                save_path = model_path + f"/chkpnt{iteration}.pth"
                torch.save((gaussians.capture(), iteration), save_path)
                
            if iteration in testing_iterations:
                psnr_test = evaluate_psnr(scene, render_finetune, {"pipe": pipe, "bg_color": background, "opt": opt, "base_color_scale": base_color_scale}, iteration, envname)

            if tb_writer:
                for key, value in tb_dict.items():
                    tb_writer.add_scalar(key, value, iteration)
        iteration += 1

    # test_transforms_file = os.path.join(args.source_path, "transforms_test.json")
    # contents = load_json_config(test_transforms_file)

    # fovx = contents["camera_angle_x"]
    # frames = contents["frames"]

    # envname = os.path.splitext(os.path.basename(envmap_path))[0]
    # render_kwargs = {
    #     "pc": gaussians,
    #     "pipe": pipe,
    #     "bg_color": background,
    #     "training": False,
    #     "relight": True,
    #     "base_color_scale": base_color_scale,
    # }
    # psnr_pbr = 0.0
    # ssim_pbr = 0.0
    # lpips_pbr = 0.0

    # for idx, frame in enumerate(tqdm(frames, leave=False)):
    #     image_path = os.path.join(args.source_path, f"test_{idx:03}/" + frame["file_path"].split("/")[-1] + "_" + envname + ".png")
    #     # NeRF 'transform_matrix' is a camera-to-world transform
    #     c2w = np.array(frame["transform_matrix"])
    #     # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    #     c2w[:3, 1:3] *= -1

    #     # get the world-to-camera transform and set R, T
    #     w2c = np.linalg.inv(c2w)
    #     R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    #     T = w2c[:3, 3]

    #     image_rgba = load_img_rgb(image_path)
    #     image = image_rgba[..., :3]
    #     mask = image_rgba[..., 3:]
    #     gt_image = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
    #     mask = torch.from_numpy(mask).permute(2, 0, 1).float().cuda()
    #     gt_image = gt_image * mask + bg * (1 - mask)
        
    #     H = image.shape[0]
    #     W = image.shape[1]
    #     fovy = focal2fov(fov2focal(fovx, W), H)

    #     custom_cam = Camera(colmap_id=0, R=R, T=T,
    #                         FoVx=fovx, FoVy=fovy,
    #                         image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name=None, uid=0)

def set_gaussian_para(gaussians, opt):
    gaussians.init_base_color_value = opt.init_base_color_value
    gaussians.init_metallic_value = opt.init_metallic_value
    gaussians.init_roughness_value = opt.init_roughness_value

def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration, base_color_scale=None):
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, opt=opt, base_color_scale=base_color_scale)

        error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])

        visualization_list = [
            viewpoint_cam.original_image.cuda(),
            render_pkg["render"],
            render_pkg["diffuse"],
            render_pkg["specular"],
            render_pkg["render_sh"],
            render_pkg["base_color_linear"],
            render_pkg["base_color"],
            render_pkg["roughness"].repeat(3, 1, 1),
            render_pkg["visibility"].repeat(3, 1, 1),
            render_pkg["light_indirect"],
            render_pkg["light_direct"],
            render_pkg["light"],
            render_pkg["render_direct"],
            render_pkg["render_indirect"],
            render_pkg["rend_alpha"].repeat(3, 1, 1),
            visualize_depth(render_pkg["surf_depth"]),
            render_pkg["rend_normal"] * 0.5 + 0.5,
            render_pkg["surf_normal"] * 0.5 + 0.5,
            error_map,
            render_pkg["render_env"],
            render_pkg["render_radiosity"].repeat(3,1,1)
        ]
            
        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 1600
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

        env_dict = gaussians.render_env_map()

        grid = [
            rgb_to_srgb(env_dict["env1"].permute(2, 0, 1)),
            rgb_to_srgb(env_dict["env2"].permute(2, 0, 1)),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))
        torch.cuda.empty_cache()

      
NORM_CONDITION_OUTSIDE = False
def prepare_output_and_logger():    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def evaluate_psnr(scene, renderFunc, renderkwargs, iteration, envname=""):    
    eval_path = os.path.join(scene.model_path, "eval", "ours_{}".format(iteration))
    os.makedirs(eval_path, exist_ok=True)
    pbr_psnr_test = 0.0
    sh_psnr_test = 0.0
    os.makedirs(os.path.join(eval_path, "pbr"), exist_ok=True)
    os.makedirs(os.path.join(eval_path, "sh"), exist_ok=True)
    os.makedirs(os.path.join(eval_path, "radiosity"), exist_ok=True)
    if len(scene.getTestCameras()):
        for idx, viewpoint in enumerate(tqdm(scene.getTestCameras())):
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            pbr_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            sh_image = torch.clamp(render_pkg["render_sh"], 0.0, 1.0)
            opacity = render_pkg["render_alpha"]
            radiosity = render_pkg["render_radiosity"] * 2.0
            # gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            # gt_image = 
            gt_relight_path = viewpoint.image_path.replace(".png", "_" + envname + ".png")
            if os.path.exists(gt_relight_path) and not ("mipnerf" in gt_relight_path):
                gt_image_np = load_img_rgb(gt_relight_path)
                gt_image = torch.from_numpy(gt_image_np[..., :3] * gt_image_np[..., 3:4]).permute(2, 0, 1).float().cuda()
                pbr_psnr_test += psnr(pbr_image, gt_image).mean().double()
                sh_psnr_test += psnr(sh_image, gt_image).mean().double()
            pbr_image = pbr_image + (1-opacity)
            sh_image = sh_image + (1-opacity)
            save_image(pbr_image, os.path.join(eval_path, "pbr", '{0:05d}'.format(idx) + ".png"))
            save_image(sh_image, os.path.join(eval_path, "sh", '{0:05d}'.format(idx) + ".png"))
            save_image(radiosity, os.path.join(eval_path, "radiosity", '{0:05d}'.format(idx) + ".png"))
            # save_image(torch.clamp(render_pkg["diffuse"], 0.0, 1.0), os.path.join(eval_path, '{0:05d}_diffuse'.format(idx) + ".png"))
            # save_image(torch.clamp(render_pkg["specular"], 0.0, 1.0), os.path.join(eval_path, '{0:05d}_specular'.format(idx) + ".png"))
        pbr_psnr_test /= len(scene.getTestCameras())
        sh_psnr_test /= len(scene.getTestCameras())
        print("\n[ITER {}] Evaluating test set: PBR_PSNR {} NVS_PSNR {}".format(iteration, pbr_psnr_test, sh_psnr_test))
        with open(os.path.join(eval_path, "psnr.txt"), 'w') as psnr_f:
            psnr_f.write(str(pbr_psnr_test) + "\n" + str(sh_psnr_test))
    torch.cuda.empty_cache()
    return pbr_psnr_test, sh_psnr_test

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000,60000,70000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("-c", "--start_checkpoint", type=str, default = None)
    parser.add_argument("--start_checkpoint_refgs", type=str, default = None)
    parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    parser.add_argument('--envmap_path', type=str, default="")
    parser.add_argument("--albedo_rescale", default=2, type=int, help="0: no scale; 1: single channel scale; 2: three channel scale")
    parser.add_argument("--reset_sh", action="store_true", help="reset SH to 0")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.save_iterations = args.save_iterations + [i for i in range(1, args.iterations+1, 500)]
    args.checkpoint_iterations = args.checkpoint_iterations + [i for i in range(5000, args.iterations+1, 5000)]
    
    # Set up output folder
    os.makedirs(args.model_path, exist_ok = True)
    full_cmd = f"python {' '.join(sys.argv)}"
    print("Command: " + full_cmd)
    
    with open(os.path.join(args.model_path, "cmd.txt"), 'w') as cmd_f:
        cmd_f.write(full_cmd)
    
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    print("Output folder: {}".format(args.model_path))
    args.visualize_path = os.path.join(args.model_path, "visualize")
    os.makedirs(args.visualize_path, exist_ok=True)
    print("Visualization folder: {}".format(args.visualize_path))
    

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint_refgs, args.model_path, args.envmap_path, args)

    # All done
    print("\nTraining complete.")