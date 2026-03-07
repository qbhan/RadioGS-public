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
from utils.loss_utils import calculate_loss, l1_loss, calculate_loss2, calculate_loss4
from gaussian_renderer import render_surfel, render_initial, render_volume
import sys
from scene import Scene, RefGaussianModel as GaussianModel
from utils.general_utils import safe_state
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments.refgs import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from utils.image_utils import visualize_depth
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from utils.graphics_utils import rgb_to_srgb
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger()

    # Set up parameters 
    TOT_ITER = opt.iterations + 1
    TEST_INTERVAL = 1000
    MESH_EXTRACT_INTERVAL = 2000

    # For real scenes
    USE_ENV_SCOPE = opt.use_env_scope  # False
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        METALLIC_MSK_LOSS_W = 0.4

    gaussians = GaussianModel(dataset.sh_degree)
    set_gaussian_para(gaussians, opt, vol=(opt.volume_render_until_iter > opt.init_until_iter)) # #
    scene = Scene(dataset, gaussians)  # init all parameters(pos, scale, rot...) from pcds
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussExtractor = GaussianExtractor(gaussians, render_initial, pipe, bg_color=bg_color) 

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_normal_smooth_for_log = 0.0
    ema_depth_smooth_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_test = 0

    progress_bar = tqdm(range(first_iter, TOT_ITER), desc="Training progress")
    first_iter += 1
    iteration = first_iter

    print(f'Propagation until: {opt.normal_prop_until_iter }')
    print(f'Densify until: {opt.densify_until_iter}')
    print(f'Total iterations: {TOT_ITER}')

    initial_stage = opt.initial
    if not initial_stage:
        opt.init_until_iter = 0

    # Training loop
    while iteration < TOT_ITER:
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Increase SH levels every 1000 iterations
        if iteration > opt.feature_rest_from_iter and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Control the init stage
        if iteration > opt.init_until_iter:
            initial_stage = False
        
        # Control the indirect stage
        if iteration == opt.indirect_from_iter + 1:
            opt.indirect = 1

        if iteration == (opt.volume_render_until_iter + 1) and opt.volume_render_until_iter > opt.init_until_iter:
            reset_gaussian_para(gaussians, opt)

        # Initialize envmap
        if not initial_stage:
            if iteration <= opt.volume_render_until_iter:
                envmap2 = gaussians.get_envmap_2 
                envmap2.build_mips()
            else:
                envmap = gaussians.get_envmap_1
                envmap.build_mips()

        # Control the radiosity stage
        if iteration == opt.radiosity_from_iter+1:
            opt.radiosity = 1

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Set render
        # print('initial_stage:', initial_stage, 'volume_render_until_iter:', opt.volume_render_until_iter)
        render = select_render_method(iteration, opt, initial_stage)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, srgb=pipe.srgb, opt=opt, training=True, iteration=iteration)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        total_loss, tb_dict, radiosity_tensor = calculate_loss(viewpoint_cam, gaussians, render_pkg, opt, iteration)
        if radiosity_tensor is not None:
            with torch.no_grad():
                # print("Radiosity stats - min:", radiosity.min().item(), "max:", radiosity.max().item(), "mean:", radiosity.mean().item(), "std:", radiosity.std().item())
                tb_dict["radiosity_min"] = radiosity_tensor.min().item()
                tb_dict["radiosity_max"] = radiosity_tensor.max().item()
                tb_dict["radiosity_mean"] = radiosity_tensor.mean().item()
                tb_dict["radiosity_std"] = radiosity_tensor.std().item()
        dist_loss, normal_loss, loss, Ll1, normal_smooth_loss, depth_smooth_loss = tb_dict["loss_dist"], tb_dict["loss_normal_render_depth"], tb_dict["loss0"], tb_dict["loss_l1"], tb_dict["loss_normal_smooth"], tb_dict["loss_depth_smooth"] 

        def get_outside_msk():
            return None if not USE_ENV_SCOPE else torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2
        
        if USE_ENV_SCOPE and 'metallic_map' in render_pkg:
            metallics = gaussians.get_metallic
            metallic_msk_loss = metallics[get_outside_msk()].mean()
            total_loss += METALLIC_MSK_LOSS_W * metallic_msk_loss
        
        total_loss.backward()

        iter_end.record()
        with torch.no_grad():
            
            if iteration % TEST_INTERVAL == 0 or iteration == first_iter + 1 or iteration == opt.volume_render_until_iter + 1:
                save_training_vis(viewpoint_cam, gaussians, background, render, pipe, opt, iteration, initial_stage)

            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log
            ema_normal_smooth_for_log = 0.4 * normal_smooth_loss + 0.6 * ema_normal_smooth_for_log
            ema_depth_smooth_for_log = 0.4 * depth_smooth_loss + 0.6 * ema_depth_smooth_for_log
            ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Distort": f"{ema_dist_for_log:.{5}f}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "PSNR-train": f"{ema_psnr_for_log:.{4}f}",
                    "PSNR-test": f"{psnr_test:.{4}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == TOT_ITER:
                progress_bar.close()

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

            if iteration in testing_iterations:
                psnr_test = evaluate_psnr(scene, render, {"pipe": pipe, "bg_color": background, "opt": opt, "srgb":pipe.srgb}, iteration)
                
            # Densification
            if iteration < opt.densify_until_iter and iteration != opt.volume_render_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, radiosity_tensor)

                if iteration <= opt.init_until_iter:
                    opacity_reset_intval = 3000
                    densification_interval = 100
                elif iteration <= opt.normal_prop_until_iter :
                    opacity_reset_intval = 3000
                    densification_interval = opt.densification_interval_when_prop
                else:
                    opacity_reset_intval = 3000
                    densification_interval = 100

                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent,
                                                size_threshold)

                HAS_RESET0 = False
                if iteration % opacity_reset_intval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    gaussians.reset_opacity_mask0()
                    gaussians.reset_metallic_mask(exclusive_msk=outside_msk)
                    if opt.reset_sh_features:
                        gaussians.reset_features()
                if opt.opac_lr0_interval > 0 and (
                        opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.opac_lr0_interval == 0:
                    gaussians.set_opacity_lr(opt.opacity_lr)
                if (opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.normal_prop_interval == 0:
                    if not HAS_RESET0:
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity_mask1(exclusive_msk=outside_msk)
                        if opt.reset_sh_features:
                            gaussians.reset_features()
                        if iteration > opt.volume_render_until_iter and opt.volume_render_until_iter > opt.init_until_iter:
                            gaussians.dist_color(exclusive_msk=outside_msk)

                        gaussians.reset_scale(exclusive_msk=outside_msk)
                        if opt.opac_lr0_interval > 0 and iteration != opt.normal_prop_until_iter :
                            gaussians.set_opacity_lr(0.0)
                
            if (iteration >= opt.indirect_from_iter and iteration % MESH_EXTRACT_INTERVAL == 0) or iteration == (opt.indirect_from_iter):
                if not HAS_RESET0:
                    gaussExtractor.reconstruction(scene.getTrainCameras())
                    if 'ref_real' in dataset.source_path:
                        mesh = gaussExtractor.extract_mesh_unbounded(resolution=opt.mesh_res)
                    else:
                        depth_trunc = (gaussExtractor.radius * 2.0) if opt.depth_trunc < 0  else opt.depth_trunc
                        voxel_size = (depth_trunc / opt.mesh_res) if opt.voxel_size < 0 else opt.voxel_size
                        sdf_trunc = 5.0 * voxel_size if opt.sdf_trunc < 0 else opt.sdf_trunc
                        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                    mesh = post_process_mesh(mesh, cluster_to_keep=opt.num_cluster)
                    # ply_path = os.path.join(model_path,f'test_{iteration:06d}.ply')
                    # o3d.io.write_triangle_mesh(ply_path, mesh)
                    gaussians.update_mesh(mesh)

            if iteration < TOT_ITER:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if tb_writer and iteration % 10 == 0: # Log to TensorBoard only one tenth a time for memory
                for key, value in tb_dict.items():
                    tb_writer.add_scalar(key, value, iteration)

        iteration += 1

def select_render_method(iteration, opt, initial_stage):
    if initial_stage:
        render = render_initial
    elif iteration <= opt.volume_render_until_iter:
        render = render_volume
    else:   
        render = render_surfel
    return render

def set_gaussian_para(gaussians, opt, vol=False):
    gaussians.enlarge_scale = opt.enlarge_scale
    gaussians.rough_msk_thr = opt.rough_msk_thr 
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.init_metallic_value = opt.init_metallic_value
    gaussians.metallic_msk_thr = opt.metallic_msk_thr

def reset_gaussian_para(gaussians, opt):
    gaussians.reset_base_color()
    gaussians.reset_metallic(opt.init_metallic_value)
    gaussians.reset_roughness(opt.init_roughness_value)
    gaussians.metallic_msk_thr = opt.metallic_msk_thr
    gaussians.rough_msk_thr = opt.rough_msk_thr

def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration, initial_stage):
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, srgb=pipe.srgb, opt=opt)

        error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])

        if initial_stage:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),
                render_pkg["render"], 
                render_pkg["rend_alpha"].repeat(3, 1, 1),
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5, 
                render_pkg["surf_normal"] * 0.5 + 0.5, 
                error_map 
            ]

        elif iteration <= opt.volume_render_until_iter:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  # (0,0)
                render_pkg["render"], # (0,1)
                render_pkg["base_color_map"],  # (0,2)
                render_pkg["diffuse_map"],     # (0,3)
                render_pkg["specular_map"],  # (1,0)
                render_pkg["metallic_map"].repeat(3, 1, 1),  # (1,1)
                render_pkg["roughness_map"].repeat(3, 1, 1), # (1,2)
                render_pkg["rend_alpha"].repeat(3, 1, 1),  # (1,3)
                visualize_depth(render_pkg["surf_depth"]), # (2,0)
                render_pkg["rend_normal"] * 0.5 + 0.5,  # (2,1)
                render_pkg["surf_normal"] * 0.5 + 0.5, # (2,2)
                error_map # (2,3)
            ]
            if opt.indirect:
                visualization_list += [
                    render_pkg["visibility"].repeat(3, 1, 1), # (3,0)
                    render_pkg["direct_light"], # (3,1)
                    render_pkg["indirect_light"], # (3,2)
                ]

        else:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"],  
                render_pkg["base_color_map"],  
                render_pkg["diffuse_map"],
                render_pkg["specular_map"],
                render_pkg["metallic_map"].repeat(3, 1, 1),  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5,  
                render_pkg["surf_normal"] * 0.5 + 0.5,  
                error_map, 
            ]

        if 'render_sh' in render_pkg:
            visualization_list += [
                render_pkg["render_sh"],
            ]
            error_map_sh = torch.abs(render_pkg["render"] - render_pkg["render_sh"]) * 10
            visualization_list += [
                error_map_sh,
            ]
        
        if 'render_rad' in render_pkg:
            visualization_list += [
                render_pkg["render_rad"].repeat(3, 1, 1),
            ]

        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 800
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

        if not initial_stage:
            if opt.volume_render_until_iter > opt.init_until_iter and iteration <= opt.volume_render_until_iter:
                env_dict = gaussians.render_env_map_2() 
            else:
                env_dict = gaussians.render_env_map_1()

            if pipe.srgb:
                grid = [
                    rgb_to_srgb(env_dict["env1"].permute(2, 0, 1)),
                    rgb_to_srgb(env_dict["env2"].permute(2, 0, 1)),
                ]
            else:
                grid = [
                    env_dict["env1"].permute(2, 0, 1),
                    env_dict["env2"].permute(2, 0, 1),
                ]
            grid = make_grid(grid, nrow=1, padding=10)
            save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))
      
def prepare_output_and_logger():    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    args.visualize_path = os.path.join(args.model_path, "visualize")
    
    os.makedirs(args.visualize_path, exist_ok=True)
    print("Visualization folder: {}".format(args.visualize_path))
    
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
    torch.cuda.empty_cache()
    if len(scene.getTestCameras()):
        for idx, viewpoint in enumerate(tqdm(scene.getTestCameras())):
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()
            
            save_image(image, os.path.join(eval_path, '{0:05d}'.format(idx) + ".png"))
            if 'render_rad' in render_pkg:  
                render_rad = torch.clamp(render_pkg['render_rad'], 0.0, 1.0).repeat(3, 1, 1)
                # save_image(render_rad, os.path.join(eval_path, 'rad_{0:05d}'.format(idx) + ".png"))
                render_rad *= 2.0
                save_image(torch.clamp(render_rad, 0.0, 1.0), os.path.join(eval_path, 'rad2_{0:05d}'.format(idx) + ".png"))
                # render_rad *= 2.5
                # save_image(torch.clamp(render_rad, 0.0, 1.0), os.path.join(eval_path, 'rad5_{0:05d}'.format(idx) + ".png"))
                # render_rad *= 2.0
                # save_image(torch.clamp(render_rad, 0.0, 1.0), os.path.join(eval_path, 'rad10_{0:05d}'.format(idx) + ".png"))
            # if 'contrib' in render_pkg:
            #     if render_pkg['contrib'].mean() == 0.0:
            #         continue
            #     # print( render_pkg['contrib'].max(), render_pkg['contrib'].min())
            #     # contrib = render_pkg['contrib'] / (render_pkg['contrib'].max() + 1e-5)
            #     contrib = torch.clamp(render_pkg['contrib'], 0.0, 1.0)
            #     save_image(contrib.repeat(3, 1, 1), os.path.join(eval_path, 'contrib_{0:05d}'.format(idx) + ".png"))
            if 'render_sh' in render_pkg:
                render_sh = torch.clamp(render_pkg['render_sh'], 0.0, 1.0)
                save_image(render_sh, os.path.join(eval_path, 'sh_{0:05d}'.format(idx) + ".png"))
                # error_map_sh = torch.clamp(torch.abs(image - render_sh), 0.0, 1.0)
                # save_image(error_map_sh, os.path.join(eval_path, 'errsh_{0:05d}'.format(idx) + ".png"))
                error_map_sh = torch.clamp(torch.abs(image - render_sh)*2.0, 0.0, 1.0)
                save_image(error_map_sh, os.path.join(eval_path, 'errsh2_{0:05d}'.format(idx) + ".png"))
                # error_map_sh = torch.clamp(torch.abs(image - render_sh)*5.0, 0.0, 1.0)
                # save_image(error_map_sh, os.path.join(eval_path, 'errsh5_{0:05d}'.format(idx) + ".png"))
                # error_map_sh = torch.clamp(torch.abs(image - render_sh)*10.0, 0.0, 1.0)
                # save_image(error_map_sh, os.path.join(eval_path, 'errsh10_{0:05d}'.format(idx) + ".png"))
            if 'rend_normal' in render_pkg:
                rend_normal = render_pkg['rend_normal'] * 0.5 + 0.5
                save_image(rend_normal, os.path.join(eval_path, 'normal_{0:05d}'.format(idx) + ".png"))

        psnr_test /= len(scene.getTestCameras())
        psnr_txt = "\n[ITER {}] Evaluating test set: PSNR {}, # Gaussian {}".format(iteration, psnr_test, scene.gaussians.get_xyz.shape[0])
        print(psnr_txt)
        with open(os.path.join(eval_path, "psnr.txt"), 'w') as psnr_f:
            psnr_f.write(str(psnr_txt))
    torch.cuda.empty_cache()
    return psnr_test

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,20000,30000,40000,50000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.checkpoint_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)
    # if args.initial == 0 and args.train_sh_vol :
    #     args.test_iterations = args.test_iterations + [i for i in range(2500, args.densify_until_iter, 1000)]
    #     args.test_iterations = args.test_iterations + [i for i in range(args.densify_until_iter, args.iterations+1, 5000)]
    # elif args.initial == 1:
    args.test_iterations = args.test_iterations + [i for i in range(10000, args.iterations+1, 10000)]
    args.test_iterations.append(args.volume_render_until_iter)

    if not args.model_path:
        current_time = datetime.now().strftime('%m%d_%H%M')
        last_subdir = os.path.basename(os.path.normpath(args.source_path))
        args.model_path = os.path.join(
            "./output/", f"{last_subdir}/",
            f"{last_subdir}-{current_time}"
        )
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")