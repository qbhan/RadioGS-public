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

import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_radiogs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from arguments.refgs import ModelParams as RefGSModelParams, PipelineParams as RefGSPipelineParams
from scene import GaussianModel, RadioGSModel, RefGaussianModel
from scene.dataset_readers import load_img_rgb
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
import json
import matplotlib.pyplot as plt
from gaussian_renderer.radiogs import sample_incident_rays
from utils.graphics_utils import rgb_to_srgb
from torchvision.utils import save_image

def plot_opacity_distribution(gaussians, save_path, title="Opacity Distribution"):
    """
    Plot the opacity distribution of Gaussians and save as an image.
    
    Args:
        gaussians: GaussianModel instance
        save_path: Path to save the plot image
        title: Title for the plot
    """
    # Get opacity values and convert to numpy
    opacity_values = gaussians.get_opacity.detach().cpu().numpy().flatten()
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.hist(opacity_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Opacity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_opacity = opacity_values.mean()
    std_opacity = opacity_values.std()
    plt.text(0.02, 0.98, f'Mean: {mean_opacity:.4f}\nStd: {std_opacity:.4f}\nCount: {len(opacity_values)}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Opacity distribution plot saved to: {save_path}")

import numpy as np
import matplotlib.pyplot as plt

def plot_opacity_bins(gaussians, save_path, title="Opacity Value Distribution (0~1 bins)"):
    """
    Plot the opacity distribution using fixed opacity value bins (e.g., 0~0.1, 0.1~0.2, ...).
    
    Args:
        gaussians: GaussianModel instance
        save_path: Path to save the plot image
        title: Title for the plot
    """
    # Get opacity values
    opacity_values = gaussians.get_opacity.detach().cpu().numpy().flatten()

    # Define fixed bins from 0.0 to 1.0 at 0.1 intervals
    bins = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
    
    # Compute histogram counts
    counts, _ = np.histogram(opacity_values, bins=bins)
    
    # Convert to ratios (%)
    ratios = counts / len(opacity_values) * 100
    
    # X-axis labels
    labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(bins)-1)]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(labels, ratios, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Opacity Range')
    plt.ylabel('Percentage of Gaussians (%)')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Add stats
    mean_opacity = opacity_values.mean()
    std_opacity = opacity_values.std()

    plt.text(0.02, 0.98,
             f"Mean: {mean_opacity:.4f}\nStd: {std_opacity:.4f}\nCount: {len(opacity_values)}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Opacity bin distribution plot saved to: {save_path}")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, extra='', only_dist=False, start=0, opt=None):
    if len(extra) > 0:
        name = name + "_" + extra
    path_prefix = os.path.join(model_path, name, "ours_{}".format(iteration))
    gts_path = os.path.join(path_prefix, "gt")
    
    keys = ["render", "render_sh", "render_env", "diffuse", "specular", "roughness", "base_color", "base_color_linear", "rend_alpha", "rend_normal", "visibility", "light_direct", "light_indirect", "render_direct", "render_indirect"]
    keys += ["render_radiosity"]

    # define render function
    render = render_radiogs

    makedirs(gts_path, exist_ok=True)
    for key in keys:
        makedirs(os.path.join(path_prefix, key), exist_ok=True)

    # Plot opacity distribution
    opacity_plot_path = os.path.join(path_prefix, f"opacity_distribution_{name}.png")
    plot_opacity_distribution(gaussians, opacity_plot_path, f"Opacity Distribution - {name}")
    plot_opacity_bins(gaussians, os.path.join(path_prefix, f"opacity_percentiles_{name}.png"), f"Opacity Percentile Distribution - {name}")

    if only_dist: 
        print("Only distribution mode enabled, skipping rendering.")
        return
    psnr_avg = 0.0
    ssim_avg = 0.0
    lpips_avg = 0.0
    psnr_sh_avg = 0.0
    ssim_sh_avg = 0.0
    lpips_sh_avg = 0.0
    render_times = []
    
    # Create tqdm with custom format
    pbar = tqdm(views, desc="Rendering progress")

    for idx, view in enumerate(pbar):
        if idx < start: continue
        if idx == start:
            # update gaussian incident information
            cam = view
            dir_pp = gaussians.get_xyz - cam.camera_center
            dir_pp_normalized = dir_pp / (torch.norm(dir_pp, dim=-1, keepdim=True) + 1e-6)
            normal_dummy = gaussians.get_normal(scaling_modifier=1.0, dir_pp_normalized=dir_pp_normalized)
            incident_directions, incident_areas = sample_incident_rays(normal_dummy, is_training=False, sample_num=pipeline.diffuse_sample_num)
            gaussians.update_incidents_directions(incident_directions, incident_areas)
            gaussians.precompute_incidents(light_t_min=pipeline.light_t_min, only_vis=False, back_culling=pipeline.back_culling)
        
        # Time the rendering
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        render_pkg = render(view, gaussians, pipeline, background, opt=opt)
        end_time.record()
        
        torch.cuda.synchronize()
        render_time = start_time.elapsed_time(end_time)  # Time in milliseconds
        render_times.append(render_time)
        
        # Update tqdm with current rendering time
        pbar.set_postfix({
            'Time': f'{render_time:.1f}ms',
            'FPS': f'{1000.0/render_time:.1f}',
            'Avg': f'{np.mean(render_times):.1f}ms'
        })
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        image_sh = torch.clamp(render_pkg["render_sh"], 0.0, 1.0)
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        # if view.mask is not None:
            # print("Applying mask to image and ground truth", gt_image.shape)
            # image = image * view.mask.float().to("cuda")
            # gt_image = gt_image * view.mask.float().to("cuda")
            # image_sh = image_sh * view.mask.float().to("cuda")
        
        psnr_avg += psnr(image, gt_image).mean().double().item()
        ssim_avg += ssim(image, gt_image).mean().double().item()
        if not args.no_lpips:
            lpips_avg += lpips(image, gt_image, net_type='vgg').mean().double().item()
        # image_sh = torch.clamp(render_pkg["render_s, 0.0, 1.0)
        # if view.mask is not None:
        #     image_sh = image_sh * view.mask.float().to("cuda")
        psnr_sh_avg += psnr(image_sh, gt_image).mean().double().item()
        ssim_sh_avg += ssim(image_sh, gt_image).mean().double().item()
        if not args.no_lpips:
            lpips_sh_avg += lpips(image_sh, gt_image, net_type='vgg').mean().double().item()

        if args.no_save:
            continue
        
        torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        for key in keys:
            out = render_pkg[key]
            if 'normal' in key:
                out = (out + 1) / 2
            if 'position' in key:
                out = (out + 1) / 2
            if out.shape[0] == 1:
                out = out.repeat(3, 1, 1)
            if "render_radiosity" in key:
                # if out.mean() < 5e-2: out = out * 10.0
                out = torch.clamp(out * 2.0, 0.0, 1.0)
            # torchvision.utils.save_image(out * view.mask.float().cuda(), os.path.join(path_prefix, key, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(out, os.path.join(path_prefix, key, '{0:05d}'.format(idx) + ".png"))
            
    psnr_avg /= len(views)
    ssim_avg /= len(views)
    lpips_avg /= len(views)
    psnr_sh_avg /= len(views)
    ssim_sh_avg /= len(views)
    lpips_sh_avg /= len(views)
    
    # Calculate average rendering time
    total_render_time = np.sum(render_times)
    # average discarding top and bottom 5% outliers
    render_times_sorted = np.sort(render_times)
    trim_count = int(0.05 * len(render_times_sorted))
    trimmed_times = render_times_sorted[trim_count:-trim_count] if len(render_times_sorted) > 2 * trim_count else render_times_sorted
    avg_render_time = np.mean(trimmed_times)
    
    results_dict = {}
    results_dict["psnr_avg"] = psnr_avg
    results_dict["ssim_avg"] = ssim_avg
    results_dict["lpips_avg"] = lpips_avg
    results_dict["psnr_sh_avg"] = psnr_sh_avg
    results_dict["ssim_sh_avg"] = ssim_sh_avg
    results_dict["lpips_sh_avg"] = lpips_sh_avg
    results_dict["avg_render_time_ms"] = avg_render_time
    results_dict["total_render_time_ms"] = total_render_time
    results_dict["fps"] = 1000.0 / avg_render_time  # Frames per second
    
    print("\n[ITER {}] Evaluating {} set: PSNR {} SSIM {} LPIPS {} PSNR_SH{}".format(iteration, name, psnr_avg, ssim_avg, lpips_avg, psnr_sh_avg))
    print("Average rendering time: {:.2f} ms ({:.2f} FPS)".format(avg_render_time, 1000.0 / avg_render_time))
    print("Total rendering time: {:.2f} ms".format(total_render_time))
    with open(os.path.join(model_path, name, "nvs_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Results saved to", os.path.join(model_path, name, "nvs_results.json"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, extra: str, only_dist: bool = False, bg_color: float  = -1.0, start=0):
    with torch.no_grad():
        gaussians = RadioGSModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.build_bvh()      
        gaussians.env_map.update_pdf()
        if scene.light_rotate:
            transform = torch.tensor([
                [0, -1, 0], 
                    [0, 0, 1], 
                    [-1, 0, 0]
                ], dtype=torch.float32, device="cuda")
            gaussians.env_map.set_transform(transform)
        # first save env map
        env_dict = gaussians.render_env_map()
        env1 = rgb_to_srgb(env_dict["env1"].permute(2, 0, 1))
        env2 = rgb_to_srgb(env_dict["env2"].permute(2, 0, 1))
        
        save_image(env1, os.path.join(dataset.model_path, "env1.png"))
        save_image(env2, os.path.join(dataset.model_path, "env2.png"))
        print(f"Environment maps saved to {dataset.model_path}")
        
        if bg_color >= 0:
            bg_color = [bg_color, bg_color, bg_color]
        else:
            bg_color = [1,1,1] if (dataset.white_background or ('white' in extra)) else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        print('Gaussian Number:', gaussians.get_xyz.shape[0])

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, extra, only_dist, start)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, extra, only_dist, start)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_save", default=False, action='store_true')
    parser.add_argument("--no_lpips", default=False, action='store_true')
    parser.add_argument("-e", "--extra", default='', type=str)
    parser.add_argument("--only_dist", default=False, action='store_true')
    parser.add_argument("--bg_color", type=float, default=-1.0)
    parser.add_argument("--start", type=int, default=0, help="Start index for rendering (useful for resuming)")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.extra, args.only_dist, args.bg_color, args.start)