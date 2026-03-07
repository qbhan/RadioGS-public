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
from utils.loss_utils import calculate_loss2, calculate_loss3
from gaussian_renderer import render_radiogs
import sys
from scene import Scene, RadioGSModel
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


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, checkpoint_refgs, model_path, debug_from=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger()

    lr_scale = opt.lr_scale
    opt.position_lr_init *= lr_scale
    opt.opacity_lr *= lr_scale
    opt.scaling_lr *= lr_scale
    opt.rotation_lr *= lr_scale
    if 'nopbr' in model_path:
        print("No PBR loss, reducing learning rate for material and light")
        opt.base_color_lr *= 0.1
        opt.roughness_lr *= 0.1
        opt.metallic_lr *= 0.1
        opt.envmap_cubemap_lr *= 0.1

    render = render_radiogs

    gaussians = RadioGSModel(dataset.sh_degree)
    set_gaussian_para(gaussians, opt)
    
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt, restart=pipe.restart)
    elif checkpoint_refgs:
        (model_params, _) = torch.load(checkpoint_refgs, weights_only=False)
        gaussians.restore_from_refgs(model_params, opt)
        
    gaussians.build_bvh()
    
    if scene.light_rotate:
        transform = torch.tensor([
            [0, -1, 0], 
            [0, 0, 1], 
            [-1, 0, 0]
        ], dtype=torch.float32, device="cuda")
        gaussians.env_map.set_transform(transform)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    total_training_time = 0.0
    
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
        incident_directions, incident_areas = sample_incident_rays(normal_dummy, is_training=pipe.radiosity_random_sample, sample_num=pipe.diffuse_sample_num)
        gaussians.update_incidents_directions(incident_directions, incident_areas)
        gaussians.precompute_incidents(light_t_min=pipe.light_t_min, only_vis=True, back_culling=pipe.back_culling)
    
    while iteration < opt.iterations + 1:
        iter_start.record()

        if iteration % opt.indirect_update_interval == 0: 
            with torch.no_grad():
                gaussians.precompute_incidents(light_t_min=pipe.light_t_min, only_vis=False, back_culling=pipe.back_culling)
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(ratio=pipe.view_ratio).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt=opt, iteration=iteration, training=True)

        gt_image = viewpoint_cam.original_image.cuda()
        
        total_loss, tb_dict = calculate_loss3(viewpoint_cam, gaussians, render_pkg, opt, iteration)
        dist_loss, normal_loss, loss = tb_dict["loss_dist"], tb_dict["loss_normal_render_depth"], tb_dict["loss"]
        total_loss.backward()
            
        iter_end.record()

        if opt.rad_render_detach == 0:
            gaussians.incident_visibility = gaussians.incident_visibility.detach()
            gaussians.incident_radiance = gaussians.incident_radiance.detach()

        with torch.no_grad():
            viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            is_densify = False

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            
            if lr_scale > 0:
                if is_densify:
                    gaussians.build_bvh()
                else:
                    gaussians.update_bvh()

            torch.cuda.synchronize()
            iteration_time = iter_start.elapsed_time(iter_end) / 1000.0  # ms to seconds
            total_training_time += iteration_time

            if iteration % 500 == 0 or iteration == first_iter + 1:
                save_training_vis(viewpoint_cam, gaussians, background, render, pipe, opt, iteration)

            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log
            image = render_pkg["render"]
            ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
            
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{gaussians.get_xyz.shape[0]}",
                    "PSNR-train": f"{ema_psnr_for_log:.{4}f}",
                    "PSNR-test": f"{psnr_test:.{4}f}"
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
                psnr_test = evaluate_psnr(scene, render_radiogs, {"pipe": pipe, "bg_color": background, "opt": opt}, iteration)
                tb_dict["psnr_test"] = psnr_test

            if tb_writer:
                for key, value in tb_dict.items():
                    tb_writer.add_scalar(key, value, iteration)
        iteration += 1

    training_time_path = os.path.join(model_path, "training_time.txt")
    with open(training_time_path, 'w') as f:
        hours = int(total_training_time // 3600)
        minutes = int((total_training_time % 3600) // 60)
        seconds = total_training_time % 60
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")
        f.write(f"Formatted: {hours}h {minutes}m {seconds:.2f}s\n")
    print(f"\nTotal training time: {hours}h {minutes}m {seconds:.2f}s")

def set_gaussian_para(gaussians, opt):
    gaussians.init_base_color_value = opt.init_base_color_value
    gaussians.init_metallic_value = opt.init_metallic_value
    gaussians.init_roughness_value = opt.init_roughness_value

def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration):
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, opt=opt)

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
def evaluate_psnr(scene, renderFunc, renderkwargs, iteration):    
    eval_path = os.path.join(scene.model_path, "eval", "ours_{}".format(iteration))
    os.makedirs(eval_path, exist_ok=True)
    psnr_test = 0.0
    if len(scene.getTestCameras()):
        for idx, viewpoint in enumerate(tqdm(scene.getTestCameras())):
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()
        psnr_test /= len(scene.getTestCameras())
        print("\n[ITER {}] Evaluating test set: PSNR {}".format(iteration, psnr_test))
        with open(os.path.join(eval_path, "psnr.txt"), 'w') as psnr_f:
            psnr_f.write(str(psnr_test))
    torch.cuda.empty_cache()
    return psnr_test

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
    parser.add_argument('--no_save', action='store_true', default=False, help="do not save outputs")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    if not args.no_save:
        args.save_iterations = args.save_iterations + [i for i in range(5000, args.iterations+1, 5000)]
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
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.start_checkpoint_refgs, args.model_path)

    # All done
    print("\nTraining complete.")