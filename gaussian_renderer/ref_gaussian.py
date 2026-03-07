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
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from diff_surfel_sdf_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettings_sdf, GaussianRasterizer as GaussianRasterizer_sdf
from scene.ref_gaussian_model import RefGaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from utils.refl_utils import  get_specular_color_surfel, get_full_color_volume, get_full_color_volume_indirect, get_specular_color_surfel2
from utils.graphics_utils import rgb_to_srgb, srgb_to_rgb
import numpy as np

def load_rasterizer(pipe, 
                    image_height,
                    image_width,
                    tanfovx,
                    tanfovy,
                    bg,
                    scale_modifier,
                    viewmatrix,
                    projmatrix,
                    sh_degree,
                    campos,
                    prefiltered=False,
                    ):
    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scale_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=sh_degree,
        campos=campos,
        prefiltered=prefiltered,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return rasterizer


def compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe):
    # 2DGS normal and regularizations
    # additional regularizations
    render_alpha = allmap[1:2]
    
    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    
    # pseudo surface attributes
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()
    
    return {
        'render_alpha': render_alpha,
        'render_normal': render_normal,
        'render_depth_median': render_depth_median,
        'render_depth_expected': render_depth_expected,
        'render_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal
    }



def render_initial(viewpoint_camera, pc : RefGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None, **kwargs):
    training = kwargs.get('training', False)
    iteration = kwargs.get('iteration', 0)
    # print(training)
    use_random_color = training
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None


    if override_color is None:
        if pipe.convert_SHs_python or (pipe.bf_random):
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            normal = pc.get_normal(scaling_modifier, dir_pp_normalized)
            normal = normal / (normal.norm(dim=1, keepdim=True).clamp_min(1e-6))
            NoV = (normal * (-dir_pp_normalized)).sum(dim=1, keepdim=True)
            mask = torch.where(NoV > 0, 1.0, 0.0)
            # add random noise to backfacing splats
            if pipe.bf_random:
                if training and iteration > opt.backface_random_from_iter:
                    mask2 = (torch.rand_like(mask) < 0.3).float()
                    colors_precomp = colors_precomp * mask + torch.rand_like(colors_precomp) * (1-mask) * mask2
                else:
                    colors_precomp = colors_precomp * mask
            
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        
    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_depth_median = regularizations['render_depth_median']
    render_depth_expected = regularizations['render_depth_expected']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    if srgb: 
        rendered_image = rgb_to_srgb(rendered_image)
    final_image = rendered_image + bg_color[:, None, None] * (1 - render_alpha)

    rets =  {"render": final_image,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        'contrib': contrib[:1],
    }

    return rets




def render_surfel(viewpoint_camera, pc : RefGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None, **kwargs):
    training = kwargs.get('training', True)
    iteration = kwargs.get('iteration', 0)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    metallic = pc.get_metallic
    base_color = pc.get_base_color
    roughness = pc.get_rough

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None


    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    
    normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
    
    w_o = -dir_pp_normalized
    reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
    shs_indirect = pc.get_indirect.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    sh2indirect = eval_sh(3, shs_indirect, reflection)
    indirect = torch.clamp_min(sh2indirect, 0.0)

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = torch.cat((metallic, roughness, base_color, indirect), dim=-1),
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )

    base_color = rendered_image
    metallic = rendered_features[:1]
    roughness = rendered_features[1:2]
    albedo = rendered_features[2:5]
    indirect_light = rendered_features[5:8]

    # 2DGS normal and regularizations
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)
    
    if (opt is not None and opt.indirect) or (opt is None):
        specular, extra_dict = get_specular_color_surfel(pc.get_envmap_1, albedo.permute(1,2,0), viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), metallic=metallic.permute(1,2,0), roughness=roughness.permute(1,2,0), pc=pc, surf_depth=surf_depth, indirect_light=indirect_light.permute(1,2,0))
    else:
        specular, extra_dict = get_specular_color_surfel(pc.get_envmap_1, albedo.permute(1,2,0), viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), metallic=metallic.permute(1,2,0), roughness=roughness.permute(1,2,0), pc=pc, surf_depth=surf_depth)

    # Integrate the final image
    final_image = (1-metallic) * base_color + specular 

    if srgb:
        final_image = rgb_to_srgb(final_image)

    
    final_image = final_image + bg_color[:, None, None] * (1 - render_alpha)
    if (opt is not None and opt.indirect) or (opt is None):
        indirect_color = (1-metallic) * base_color + extra_dict['indirect_color']
        indirect_color = indirect_color + bg_color[:, None, None] * (1 - render_alpha)
        if srgb: 
            indirect_color = rgb_to_srgb(indirect_color)
        extra_dict['indirect_color'] = indirect_color
    if srgb:
        specular = rgb_to_srgb(specular)
        albedo = rgb_to_srgb(albedo)
        diffuse = rgb_to_srgb((1-metallic) * base_color)
    else:
        diffuse = (1-metallic) * base_color

    # render with sh 
    shs = pc.get_features
    colors_precomp = None
    contrib, rendered_image_sh, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    if srgb: 
        final_rendered_image_sh = rgb_to_srgb(rendered_image_sh)
    else:
        final_rendered_image_sh = rendered_image_sh
    final_image_sh = final_rendered_image_sh + bg_color[:, None, None] * (1 - render_alpha)

        
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results =  {"render": final_image,
            "metallic_map": metallic,
            "diffuse_map": diffuse,
            "specular_map": specular,
            "base_color_map": albedo,
            "roughness_map": roughness,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'contrib': contrib[:1],
    }
    results['render_sh'] = final_image_sh

    mask = render_alpha[0] > 0
    results['colors_sh'] = rendered_image_sh.permute(1, 2, 0)[mask]
    results['colors_pbr'] = final_image.permute(1, 2, 0)[mask]
    
    if (opt is not None and opt.indirect) or (opt is None):
        results.update(extra_dict)

    return results


def render_surfel2(viewpoint_camera, pc : RefGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None, **kwargs):
    training = kwargs.get('training', True)
    iteration = kwargs.get('iteration', 0)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    metallic = pc.get_metallic
    base_color = pc.get_base_color
    roughness = pc.get_rough

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None


    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    
    normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
    
    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = torch.cat((metallic, roughness, base_color), dim=-1),
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )

    base_color = rendered_image
    metallic = rendered_features[:1]
    roughness = rendered_features[1:2]
    albedo = rendered_features[2:5]

    # 2DGS normal and regularizations
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)
    
    specular = get_specular_color_surfel2(pc.get_envmap, albedo.permute(1,2,0), viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), metallic=metallic.permute(1,2,0), roughness=roughness.permute(1,2,0), pc=pc, surf_depth=surf_depth)
    
    # Integrate the final image
    final_image = (1-metallic) * base_color + specular * metallic
    
    final_image = rgb_to_srgb(final_image)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results =  {
        "render": final_image,
        "render_sh": final_image,
        "render_env": final_image,
        "diffuse": (1-metallic) * base_color,
        "specular": specular * metallic,
        "base_color": albedo,
        "base_color_linear": rgb_to_srgb(albedo),
        "roughness": roughness,
        "metallic": metallic,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        "visibility": torch.ones_like(specular[:1]),
        "light": rgb_to_srgb(specular),
        "light_indirect": rgb_to_srgb(specular),
        "light_direct": rgb_to_srgb(specular),
        'contrib': contrib,
    }
    
    return results





def render_volume(viewpoint_camera, pc : RefGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt = None, **kwargs):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    training = kwargs.get('training', False)
    iteration = kwargs.get('iteration', 0)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    metallic = pc.get_metallic
    base_color = pc.get_base_color
    roughness = pc.get_rough

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None

    dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

    normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
    
    w_o = -dir_pp_normalized
    reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
    shs_indirect = pc.get_indirect.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    sh2indirect = eval_sh(3, shs_indirect, reflection)
    indirect = torch.clamp_min(sh2indirect, 0.0)

    if opt.indirect:
        diffuse, specular, extra = get_full_color_volume_indirect(pc.get_envmap_2, means3D, base_color, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normals.contiguous(), opacity, metallic=metallic, roughness=roughness, pc=pc, indirect_light=indirect)
        visibility = extra['visibility']
        direct_light = extra["direct_light"]
    else: 
        diffuse, specular, extra = get_full_color_volume(pc.get_envmap_2, means3D, base_color, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normals.contiguous(), opacity, metallic=metallic, roughness=roughness)
    light = extra['light']
    colors_pbr = specular + diffuse

    # add random noise to backfacing splats
    if pipe.bf_random:
        NoV = (normals * (-dir_pp_normalized)).sum(-1, keepdim=True)
        mask = torch.where(NoV > 0, 1.0, 0.0)
        colors_pbr = colors_pbr * mask
        if training and iteration > opt.backface_random_from_iter:
            mask2 = (torch.rand_like(mask) < 0.3).float()
            colors_precomp = colors_pbr + torch.rand_like(colors_pbr) * (1-mask) * mask2
        else:
            colors_precomp = colors_pbr
    else:
        colors_precomp = colors_pbr

    if opt.indirect:
        features = torch.cat((roughness, metallic, diffuse, specular, base_color, light, visibility, indirect, direct_light,), dim=-1)
    else:
        features = torch.cat((roughness, metallic, diffuse, specular, base_color, light), dim=-1)

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )
    contrib_float = contrib.clone().float()[:1]
    
    # get rendered diffuse color and other paras
    full_color = rendered_image     # (3,H,W)
    render_roughness = rendered_features[:1]   # (1,H,W)
    render_metallic = rendered_features[1:2]   # (1,H,W)
    render_diffuse_color = rendered_features[2:5]
    render_specular_color = rendered_features[5:8]
    render_base_color = rendered_features[8:11]  #
    render_light = rendered_features[11:14]  # (3,H,W)
      
    if opt.indirect:
        render_visibility = rendered_features[14:15]
        render_indirect = rendered_features[15:18] 
        render_direct = rendered_features[18:21]

    if srgb:
        full_color = rgb_to_srgb(full_color)
        render_diffuse_color = rgb_to_srgb(render_diffuse_color)
        render_specular_color = rgb_to_srgb(render_specular_color)
        render_base_linear = render_base_color
        render_base_color = rgb_to_srgb(render_base_color)
        if opt.indirect:
            render_indirect = rgb_to_srgb(render_indirect)
            render_direct = rgb_to_srgb(render_direct)
    else: render_base_linear = render_base_color


    # 2DGS normal and regularizations
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    final_image = full_color + bg_color[:, None, None] * (1 - render_alpha)
    
    # render with sh 
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_sh = torch.clamp_min(sh2rgb + 0.5, 0.0)
    if pipe.bf_random:
        # print('rasterizer version 2, add bf random for sh!')
        NoV = (normals * (-dir_pp_normalized)).sum(-1, keepdim=True)
        mask = torch.where(NoV > 0, 1.0, 0.0)
        colors_sh = colors_sh * mask
        if training and iteration > opt.backface_random_from_iter:
            # print('add bf random for sh!')
            mask2 = (torch.rand_like(mask) < 0.3).float()
            colors_sh_bf = colors_sh + torch.rand_like(colors_sh) * (1-mask) * mask2
        else:
            colors_sh_bf = colors_sh
    else:
        colors_sh_bf = colors_sh
    # colors_sh_bf = colors_sh
    # render radiosity
    rad = torch.abs(colors_sh - colors_pbr).mean(-1, keepdim=True)
    
    rad_accum = pc.get_radiosity_accum
    contrib, rendered_image_sh, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_sh_bf,
        features = rad,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    rendered_rad = rendered_features[:1]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results =  {"render": final_image,
            "metallic_map": render_metallic,
            "diffuse_map": render_diffuse_color,
            "specular_map": render_specular_color,
            "base_color_map": render_base_color,
            "base_color_linear": render_base_linear,
            "roughness_map": render_roughness,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'light': render_light,
    }

    if opt.indirect:
        results.update(
            {
                "visibility": render_visibility,
                "indirect_light": render_indirect,
                "direct_light": render_direct
            }
        )

    if srgb:
        rendered_image_sh = rgb_to_srgb(rendered_image_sh)
    final_image_sh = rendered_image_sh + bg_color[:, None, None] * (1 - render_alpha)
    results.update(
        {
            "render_sh": final_image_sh,
        }
    )

    results.update(
        {
            "colors_sh": colors_sh,
            "colors_pbr": colors_pbr,
            "render_rad": rendered_rad,
        }
    )
    return results

def render_volume_test(viewpoint_camera, pc : RefGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt = None, base_color_scale=None, relight=False, **kwargs):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    training = kwargs.get('training', False)
    iteration = kwargs.get('iteration', 0)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
  
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    metallic = pc.get_metallic
    base_color = pc.get_base_color
    roughness = pc.get_rough

    if base_color_scale is not None:
        base_color = base_color * base_color_scale[None, :]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None

    dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

    normals = pc.get_normal(scaling_modifier, dir_pp_normalized)

    if not relight:
        w_o = -dir_pp_normalized
        reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
        shs_indirect = pc.get_indirect.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        sh2indirect = eval_sh(3, shs_indirect, reflection)
        indirect = torch.clamp_min(sh2indirect, 0.0)

    else:
        indirect = torch.zeros_like(base_color)
    diffuse, specular, extra = get_full_color_volume_indirect(pc.get_envmap_2, means3D, base_color, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normals.contiguous(), opacity, metallic=metallic, roughness=roughness, pc=pc, indirect_light=indirect)
    visibility = extra['visibility']
    direct_light = extra["direct_light"]
    colors_precomp = specular + diffuse
    light = extra['light']

    rays_d = viewpoint_camera.rays_d_hw
    env_only = pc.get_envmap_2(rays_d, mode='pure_env').permute(2,0,1)
    colors_pbr = specular + diffuse

    # add random noise to backfacing splats
    if pipe.bf_random:
        NoV = (normals * (-dir_pp_normalized)).sum(-1, keepdim=True)
        mask = torch.where(NoV > 0, 1.0, 0.0)
        colors_pbr = colors_pbr * mask
        if training and iteration > opt.backface_random_from_iter:
            # print('add bf random for pbr!')
            mask2 = (torch.rand_like(mask) < 0.3).float()
            colors_precomp = colors_pbr + torch.rand_like(colors_pbr) * (1-mask) * mask2
        else:
            colors_precomp = colors_pbr
    else:
        colors_precomp = colors_pbr

    features = torch.cat((roughness, metallic, diffuse, specular, base_color, light, visibility, indirect, direct_light,), dim=-1)

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )
    contrib_float = contrib.clone().float()[:1]

    # 2DGS normal and regularizations
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    # get rendered diffuse color and other paras
    full_color = rendered_image     # (3,H,W)
    render_roughness = rendered_features[:1]   # (1,H,W)
    render_metallic = rendered_features[1:2]   # (1,H,W)
    render_diffuse_color = rendered_features[2:5]
    render_specular_color = rendered_features[5:8]
    render_base_color = rendered_features[8:11]  #
    render_light = rendered_features[11:14]  # (3,H,W)
    render_visibility = rendered_features[14:15]
    render_indirect = rendered_features[15:18] 
    render_direct = rendered_features[18:21]
    render_env = full_color + env_only * (1 - render_alpha)

    if srgb:
        full_color = rgb_to_srgb(full_color)
        render_diffuse_color = rgb_to_srgb(render_diffuse_color)
        render_specular_color = rgb_to_srgb(render_specular_color)
        render_base_linear = render_base_color
        render_base_color = rgb_to_srgb(render_base_color)
        render_indirect = rgb_to_srgb(render_indirect)
        render_direct = rgb_to_srgb(render_direct)
        env_only = rgb_to_srgb(env_only)
        render_env = rgb_to_srgb(render_env)
    else: render_base_linear = render_base_color

    final_image = full_color + bg_color[:, None, None] * (1 - render_alpha)
    
    # render with sh 
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_sh = torch.clamp_min(sh2rgb + 0.5, 0.0)
    if pipe.bf_random:
        NoV = (normals * (-dir_pp_normalized)).sum(-1, keepdim=True)
        mask = torch.where(NoV > 0, 1.0, 0.0)
        colors_sh = colors_sh * mask
        if training and iteration > opt.backface_random_from_iter:
            mask2 = (torch.rand_like(mask) < 0.3).float()
            colors_sh_bf = colors_sh + torch.rand_like(colors_sh) * (1-mask) * mask2
        else:
            colors_sh_bf = colors_sh
    else:
        colors_sh_bf = colors_sh
    
    # render radiosity
    rad = torch.abs(colors_sh - colors_pbr).mean(-1, keepdim=True)
    
    contrib, rendered_image_sh, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_sh_bf,
        features = rad,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    rendered_rad = rendered_features[:1]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results =  {"render": final_image,
            "render_env": render_env,
            "metallic": render_metallic,
            "diffuse": render_diffuse_color,
            "specular": render_specular_color,
            "base_color": render_base_color,
            "base_color_linear": render_base_linear,
            "roughness": render_roughness,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'light': render_light,
            'env_only': env_only,
    }

    results.update(
        {
            "visibility": render_visibility,
            "light_indirect": render_indirect,
            "light_direct": render_direct
        }
    )
    if srgb:
        rendered_image_sh = rgb_to_srgb(rendered_image_sh)
    final_image_sh = rendered_image_sh + bg_color[:, None, None] * (1 - render_alpha)
    results.update(
        {
            "render_sh": final_image_sh,
        }
    )

    results.update(
        {
            "colors_sh": colors_sh,
            "colors_pbr": colors_pbr,
            "render_rad": rendered_rad,
        }
    )
    return results