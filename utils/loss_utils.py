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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from kornia.filters import spatial_gradient
from .image_utils import psnr
import numpy as np
import trimesh
import math
from utils.graphics_utils import rgb_to_srgb, srgb_to_rgb

def cos_loss(output, gt, thrsh=0, weight=1):
    cos = torch.sum(output * gt * weight, 0)
    return (1 - cos[cos < np.cos(thrsh)]).mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def my_l1_loss(pred, gt):
    return torch.abs((pred - gt))

def my_l2_loss(pred, gt):
    return ((pred - gt) ** 2)

def relMSE(pred, gt, reduction='mean'):
    num = (pred-gt)
    denom = (pred + gt) / 2 + 1e-6
    return (num / denom.detach())**2

def SMAPE(pred, gt):
    denom = (pred + gt + 1e-6)
    return torch.abs(pred - gt) / denom.detach()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()

def tv_loss(depth):
    # return spatial_gradient(data[None], order=2)[0, :, [0, 2]].abs().sum(1).mean()
    h_tv = torch.square(depth[..., 1:, :] - depth[..., :-1, :]).mean()
    w_tv = torch.square(depth[..., :, 1:] - depth[..., :, :-1]).mean()
    return h_tv + w_tv

def calculate_loss(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["rend_alpha"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"]
    visibility_filter = render_pkg["visibility_filter"]
    rend_dist = render_pkg["rend_dist"]
    gt_image = viewpoint_camera.original_image.cuda()

    Ll1 = l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    loss0 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    loss = torch.zeros_like(loss0)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()
    loss += loss0

    has_sh= (opt.train_sh_vol and iteration <= opt.volume_render_until_iter >= opt.train_sh_vol_from_iter) \
        or (opt.train_sh_surf and iteration > opt.volume_render_until_iter and iteration >= opt.train_sh_surf_from_iter)
    has_sh = has_sh and iteration > opt.init_until_iter

    if has_sh:
        rendered_image_sh = render_pkg["render_sh"]
        loss_sh = (1.0 - opt.lambda_dssim) * l1_loss(rendered_image_sh, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image_sh, gt_image))
        tb_dict["loss_sh"] = loss_sh.item()
        loss += loss_sh

    # Store radiosity tensor for densification gradient computation
    radiosity_tensor = None
    if opt.radiosity and opt.lambda_radiosity >= 0 and has_sh:
        colors_sh = render_pkg["colors_sh"]
        colors_pbr = render_pkg["colors_pbr"]
        loss_radiosity = F.l1_loss(colors_sh, colors_pbr)
        tb_dict["loss_radiosity"] = loss_radiosity.item()
        loss = loss + opt.lambda_radiosity * loss_radiosity
        # Store the tensor for gradient computation in densification
        # Use colors_pbr as it represents the physically-based rendering colors
        radiosity_tensor = torch.abs(colors_pbr - colors_sh).detach()


    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = torch.zeros_like(loss)

    if opt.lambda_normal_smooth > 0 and iteration > opt.normal_smooth_from_iter and iteration < opt.normal_smooth_until_iter:
        loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = torch.zeros_like(loss)
    
    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = torch.zeros_like(loss)
    
    if viewpoint_camera.mask is not None and opt.lambda_mask_entropy > 0:
        rendered_opacity = render_pkg["rend_alpha"]
        image_mask = viewpoint_camera.mask.float()
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy
    else:
        tb_dict["loss_mask_entropy"] = torch.zeros_like(loss)

    if opt.lambda_light > 0 and 'light' in render_pkg:
        light_direct = render_pkg["light"]
        mean_light = light_direct.mean(-1, keepdim=True).expand_as(light_direct)
        loss_light = F.l1_loss(light_direct, mean_light)
        tb_dict["loss_light"] = loss_light.item()
        loss = loss + opt.lambda_light * loss_light

    if opt.lambda_base_color_smooth > 0:
        rendered_base_color = render_pkg["base_color_linear"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        else:
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color, gt_image)
        tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

    if opt.lambda_roughness_smooth > 0:
        rendered_roughness = render_pkg["roughness_map"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        else:
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness, gt_image)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()

    if opt.lambda_light_smooth > 0:
        env = render_pkg["env_only"]
        loss_light_smooth = tv_loss(env)
        loss = loss + opt.lambda_light_smooth * loss_light_smooth

    try:
        if pc.use_sdf:
            if iteration > 1000:
                ref_dev = pc.get_invs_ref()
                loss_dev = torch.relu(ref_dev - pc.inverse_deviation)
                tb_dict['dev'] = loss_dev
                tb_dict['inv_dev'] = pc.inverse_deviation.mean().item()
                loss = loss + opt.lambda_dev * loss_dev
            if opt.lambda_proj > 0 and iteration > opt.proj_from_iteration:
                points = pc.get_shift_xyz[visibility_filter]
                points = torch.cat([points, torch.ones_like(points[:, -1:])], -1)
                points_view = points @ viewpoint_camera.world_view_transform
                points_proj = points @ viewpoint_camera.full_proj_transform
                points_depth = points_view[:, 2:3]
                uv = points_proj[:, :2] / (points_proj[:, -1:] + 1e-8)
                gaussian_proj_depth = torch.nn.functional.grid_sample(input=rendered_depth.unsqueeze(0),
                                                    grid=uv.view(1, -1, 1, 2),
                                                    mode='bilinear',
                                                    padding_mode='border'  # 'reflection', 'zeros'
                                                    )[0, 0]
                # Detach the projected depth to avoid unstable gradient.
                proj_error = torch.abs(gaussian_proj_depth.detach() - points_depth) 
                loss_proj = (proj_error * (proj_error < opt.proj_thres)).mean()
                tb_dict['proj'] = loss_proj
                loss = loss + opt.lambda_proj * loss_proj
    except:
        pass
    
        
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict, radiosity_tensor

def calculate_loss2(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_normal = render_pkg["rend_normal"]
    gt_image = viewpoint_camera.original_image.cuda()

    rendered_image = render_pkg["render"]
    Ll1 = F.l1_loss(rendered_image, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
    tb_dict["loss_l1"] = Ll1.item()
    loss = Ll1 * opt.lambda_pbr
    
    rendered_image_sh = render_pkg["render_sh"]
    loss_sh = (1.0 - opt.lambda_dssim) * l1_loss(rendered_image_sh, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image_sh, gt_image))
    loss += loss_sh * opt.lambda_nvs

    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        rend_dist = render_pkg["rend_dist"]
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = torch.zeros_like(loss)

    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        rendered_depth = render_pkg["surf_depth"]
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = torch.zeros_like(loss)
        
    if viewpoint_camera.mask is not None and opt.lambda_mask_entropy > 0:
        rendered_opacity = render_pkg["rend_alpha"]
        image_mask = viewpoint_camera.mask.float()
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy
    else:
        tb_dict["loss_mask_entropy"] = torch.zeros_like(loss)
    
    if opt.lambda_base_color_smooth > 0:
        rendered_base_color = render_pkg["base_color_linear"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        else:
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color, gt_image)
        tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth
    
    if opt.lambda_metallic_smooth > 0:
        rendered_metallic = render_pkg["metallic"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic * image_mask, gt_image)
        else:
            loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic, gt_image)
        tb_dict["loss_metallic_smooth"] = loss_metallic_smooth.item()
        loss = loss + opt.lambda_metallic_smooth * loss_metallic_smooth
    
    if opt.lambda_roughness_smooth > 0:
        rendered_roughness = render_pkg["roughness"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        else:
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness, gt_image)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth
    
    if opt.lambda_normal_smooth > 0:
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal * image_mask, gt_image)
        else:
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = torch.zeros_like(loss)
    
    if opt.lambda_light > 0:
        light_direct = render_pkg["ray_light_direct"]
        mean_light = light_direct.mean(-1, keepdim=True).expand_as(light_direct)
        loss_light = F.l1_loss(light_direct, mean_light)
        tb_dict["loss_light"] = loss_light.item()
        loss = loss + opt.lambda_light * loss_light

    if opt.lambda_light_smooth > 0:
        env = render_pkg["env_only"]
        loss_light_smooth = tv_loss(env)
        loss = loss + opt.lambda_light_smooth * loss_light_smooth
    
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict

def calculate_loss3(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_normal = render_pkg["rend_normal"]
    gt_image = viewpoint_camera.original_image.cuda()

    rendered_image = render_pkg["render"]
    # Ll1 = (1.0 - opt.lambda_dssim) * l1_loss(rendered_image, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
    Ll1 = l1_loss(rendered_image, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image))
    tb_dict["loss_l1"] = Ll1.item()
    if iteration > opt.pbr_loss_start:
        loss = Ll1 * opt.lambda_pbr
    else:
        loss = torch.zeros_like(Ll1)
    
    rendered_image_sh = render_pkg["render_sh"]
    loss_sh = (1.0 - opt.lambda_dssim) * l1_loss(rendered_image_sh, gt_image) + opt.lambda_dssim * (1.0 - ssim(rendered_image_sh, gt_image))
    loss += loss_sh * opt.lambda_nvs

    with torch.no_grad():
        rendered_image = render_pkg["render"]
        rendered_image_sh = render_pkg["render_sh"]
        psnr_pbr = psnr(rendered_image, gt_image).mean().item()
        psnr_nvs = psnr(rendered_image_sh, gt_image).mean().item()
        tb_dict["psnr_pbr"] = psnr_pbr
        tb_dict["psnr_nvs"] = psnr_nvs

    if opt.lambda_radiosity >= 0 and iteration > opt.radiosity_loss_start:
        # print('radiosity')
        if opt.rad_loss == 'l1': loss_fn = my_l1_loss
        elif opt.rad_loss == 'l2': loss_fn = my_l2_loss
        elif opt.rad_loss == 'relmse': loss_fn = relMSE
        elif opt.rad_loss == 'smape': loss_fn = SMAPE
        else: raise NotImplementedError("Unknown radiosity loss type", opt.rad_loss)

        if not opt.only_rnd_view:
            pbr_radiosity = render_pkg["pbr_radiosity"]
            nvs_radiosity = render_pkg["nvs_radiosity"]
            loss_radiosity = loss_fn(pbr_radiosity, nvs_radiosity)
        else: loss_radiosity = 0.0

        if 'rand_lhs' in render_pkg.keys() and 'rand_rhs' in render_pkg.keys():
            rand_lhs = render_pkg["rand_lhs"]
            rand_rhs = render_pkg["rand_rhs"]
            rand_loss = loss_fn(rand_lhs, rand_rhs).mean(dim=0)
            if not opt.rad_only_rndview:
                # loss_radiosity += loss_fn(rand_lhs, rand_rhs)
                loss_radiosity += rand_loss
            else:
                # loss_radiosity = loss_fn(rand_lhs, rand_rhs)
                loss_radiosity = rand_loss
            # loss_radiosity = loss_radiosity / 2.0

        if opt.weight_roughness:
            loss_radiosity = loss_radiosity * render_pkg["rad_roughness"]

        loss_radiosity = loss_radiosity.mean()

        tb_dict["loss_radiosity"] = loss_radiosity.item()
        loss = loss + opt.lambda_radiosity * loss_radiosity
        # if opt.rad_only: 
        #     tb_dict["loss"] = loss.item()
        #     return loss, tb_dict


    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        rend_dist = render_pkg["rend_dist"]
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = torch.zeros_like(loss)

    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        rendered_depth = render_pkg["surf_depth"]
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = torch.zeros_like(loss)
        
    if viewpoint_camera.mask is not None and opt.lambda_mask_entropy > 0:
        rendered_opacity = render_pkg["rend_alpha"]
        image_mask = viewpoint_camera.mask.float()
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy
    else:
        tb_dict["loss_mask_entropy"] = torch.zeros_like(loss)
    
    if opt.lambda_base_color_smooth > 0 and iteration > opt.pbr_loss_start:
        rendered_base_color = render_pkg["base_color_linear"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        else:
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color, gt_image)
        tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth
    
    if opt.lambda_metallic_smooth > 0 and iteration > opt.pbr_loss_start:
        rendered_metallic = render_pkg["metallic"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic * image_mask, gt_image)
        else:
            loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic, gt_image)
        tb_dict["loss_metallic_smooth"] = loss_metallic_smooth.item()
        loss = loss + opt.lambda_metallic_smooth * loss_metallic_smooth
    
    if opt.lambda_roughness_smooth > 0 and iteration > opt.pbr_loss_start:
        rendered_roughness = render_pkg["roughness"]
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        else:
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness, gt_image)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth
    
    if opt.lambda_normal_smooth > 0:
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal * image_mask, gt_image)
        else:
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = torch.zeros_like(loss)
    
    if opt.lambda_light > 0 and iteration > opt.pbr_loss_start:
        light_direct = render_pkg["ray_light_direct"]
        mean_light = light_direct.mean(-1, keepdim=True).expand_as(light_direct)
        loss_light = F.l1_loss(light_direct, mean_light)
        tb_dict["loss_light"] = loss_light.item()
        loss = loss + opt.lambda_light * loss_light

    if opt.lambda_light_smooth > 0 and iteration > opt.pbr_loss_start:
        env = render_pkg["env_only"]
        loss_light_smooth = tv_loss(env)
        loss = loss + opt.lambda_light_smooth * loss_light_smooth
    
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict

def calculate_loss4(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["rend_alpha"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"]
    visibility_filter = render_pkg["visibility_filter"]
    rend_dist = render_pkg["rend_dist"]
    gt_image = viewpoint_camera.original_image.cuda()

    Ll1 = l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    loss0 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    loss = torch.zeros_like(loss0)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()
    loss += loss0

    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = torch.zeros_like(loss)

    if opt.lambda_normal_smooth > 0 and iteration > opt.normal_loss_start:
        if viewpoint_camera.mask is not None:
            image_mask = viewpoint_camera.mask.float().cuda()
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal * image_mask, gt_image)
        else:
            loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = torch.zeros_like(loss)
    
    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = torch.zeros_like(loss)
    
    if viewpoint_camera.mask is not None and opt.lambda_mask_entropy > 0:
        rendered_opacity = render_pkg["rend_alpha"]
        image_mask = viewpoint_camera.mask.float()
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy
    else:
        tb_dict["loss_mask_entropy"] = torch.zeros_like(loss)
        
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict


def calculate_radiosity(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    if opt.rad_loss == 'l1': loss_fn = my_l1_loss
    elif opt.rad_loss == 'l2': loss_fn = my_l2_loss
    elif opt.rad_loss == 'relmse': loss_fn = relMSE
    elif opt.rad_loss == 'smape': loss_fn = SMAPE
    else: raise NotImplementedError("Unknown radiosity loss type", opt.rad_loss)

    pbr_radiosity = render_pkg["pbr_radiosity"]
    nvs_radiosity = render_pkg["nvs_radiosity"]
    loss_radiosity = loss_fn(pbr_radiosity, nvs_radiosity)

    if 'rand_lhs' in render_pkg.keys() and 'rand_rhs' in render_pkg.keys():
        rand_lhs = render_pkg["rand_lhs"]
        rand_rhs = render_pkg["rand_rhs"].detach()
        loss_radiosity += loss_fn(rand_lhs, rand_rhs)
        # loss_radiosity = loss_radiosity / 2.0

    if opt.weight_roughness:
        loss_radiosity = loss_radiosity * render_pkg["rad_roughness"]

    loss_radiosity = loss_radiosity.mean()

    tb_dict["loss_radiosity"] = loss_radiosity.item()
    loss = opt.lambda_radiosity * loss_radiosity
    
    tb_dict["loss"] = loss.item()

    render_pbr = render_pkg["render"].detach()
    render_nvs = render_pkg["render_sh"]

    Ll1 = l1_loss(render_nvs, render_pbr)
    ssim_val = ssim(render_nvs, render_pbr)
    loss0 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(render_nvs, render_pbr).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()
    loss += opt.lambda_nvs * loss0

    return loss, tb_dict