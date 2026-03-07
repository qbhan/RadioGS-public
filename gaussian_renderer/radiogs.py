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
from scene import RadioGSModel
from utils.point_utils import depth_to_normal, depths_to_points
from utils.graphics_utils import rotation_between_z, fibonacci_sphere_sampling, sample_ggx, ggx_pdf, random_hemisphere_sampling
from utils.refl_utils import  get_specular_color_surfel, get_full_color_volume, get_full_color_volume_indirect, get_specular_color_surfel2
from .ref_gaussian import render_initial, render_surfel, render_volume, render_surfel2
import numpy as np
from utils.system_utils import Timing
import trimesh
import nvdiffrast.torch as dr
import kornia
from torchvision.utils import save_image
from utils.point_utils import sample_gaussian_mask
from utils.sh_utils import eval_sh
from utils.general_utils import flip_align_view
from utils.graphics_utils import rgb_to_srgb, srgb_to_rgb

CHUNK_SIZE = 100000

def compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe):
    # 2DGS normal and regularizations
    # additional regularizations
    render_alpha = allmap[1:2]
    
    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median / render_alpha, 0, 0)
    
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
    
    render_var = render_depth_median - render_depth_expected.square()
    return {
        'render_alpha': render_alpha,
        'render_normal': render_normal,
        'render_depth_median': render_depth_median,
        'render_depth_expected': render_depth_expected,
        'render_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        'render_var': render_var,
    }



def render_radiogs(viewpoint_camera, pc : RadioGSModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, opt=None, iteration=-1, training=False, relight=False, base_color_scale=None, material_only=False):
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
    
    base_color = pc.get_base_color
    roughness = pc.get_rough
    
    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None
    
    shs = pc.get_features
    colors_precomp = None

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    normal = pc.get_normal(scaling_modifier=scaling_modifier, dir_pp_normalized=dir_pp_normalized)

    if base_color_scale is not None:
        base_color = base_color * base_color_scale[None, :]

    features = torch.cat([base_color, roughness], dim=-1)

    if pipe.bf_random:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        normal = normal / (normal.norm(dim=1, keepdim=True).clamp_min(1e-6))
        NoV = (normal * (-dir_pp_normalized)).sum(-1, keepdim=True)
        mask = torch.where(NoV > 0, 1.0, 0.0)
        if training and iteration > 5000:
            mask2 = (torch.rand_like(mask) < 0.3).float()
            colors_precomp = colors_precomp * mask + torch.rand_like(colors_precomp) * (1-mask) * mask2
        else:
            colors_precomp = colors_precomp * mask
        shs = None

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
    
    # 2DGS normal and regularizations
    # additional regularizations
    render_alpha = allmap[1:2]
    mask = render_alpha[0] > 0
    
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
    
    points = surf_depth.permute(1, 2, 0) * viewpoint_camera.rays_d_hw_unnormalized + viewpoint_camera.camera_center
    
    surf_normal = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    surf_normal[1:-1, 1:-1, :] = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    
    surf_normal = surf_normal.permute(2,0,1)
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()
    
    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)  
    normal_map = F.normalize(normal_map, dim=-1)

    rendered_base_color, rendered_roughness = rendered_features.split([3, 1], dim=0)

    def opacity_filter(r, m, b):
        return r * m.detach() + b * (1 - m.detach())

    if material_only:
        if bg_color.mean() > 0.0:
            rendered_roughness = rendered_roughness + bg_color[:, None, None] * (1 - render_alpha)
            rendered_base_color = rendered_base_color + bg_color[:, None, None] * (1 - render_alpha)
            rend_normal = render_normal + bg_color[:, None, None] * (1 - render_alpha)
            surf_normal = surf_normal + bg_color[:, None, None] * (1 - render_alpha)
        else: rend_normal =  render_normal
        results = {
            "roughness": rendered_roughness,
            "base_color": rgb_to_srgb(rendered_base_color),
            "base_color_linear": rendered_base_color,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            'rend_alpha': render_alpha,
            'rend_normal': rend_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
        }
        return results
    
    results = {}
    
    # calculate per-Gaussian radiometric consistency
    use_radiosity = pipe.use_radiosity
    if training and use_radiosity:
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        normals = pc.get_normal(scaling_modifier, dir_pp_normalized) # need to think whether to use the normals pped by camera view should be used

        # sample gaussians for radiosity. cannot update all gaussians since it is too expensive.
        if pipe.radiosity_gaussian_num == -1:
            sample_mask = sample_gaussian_mask(opacity, roughness, num_samples=opacity.shape[0], replacement=False)
        elif pipe.radiosity_gaussian_num > 0:
            sample_mask = sample_gaussian_mask(opacity, roughness, num_samples=pipe.radiosity_gaussian_num, replacement=False)

        # prepare the radiosity inputs based on the sample mask and gradients
        if pipe.detach_rad_mat:
            rad_base_color = pc.get_base_color[sample_mask].detach()
            rad_roughness = pc.get_rough[sample_mask].detach()
        else:
            rad_base_color = pc.get_base_color[sample_mask]
            rad_roughness = pc.get_rough[sample_mask]

        if pipe.detach_rad_normal:
            rad_normals = normals[sample_mask].detach()
            rad_points = pc.get_xyz[sample_mask].detach()
            rad_view = -dir_pp_normalized[sample_mask].detach()
        else:
            rad_normals = normals[sample_mask]
            rad_points = pc.get_xyz[sample_mask]
            rad_view = -dir_pp_normalized[sample_mask]

        # prepare SHs for radiosity lhs
        rad_shs = pc.get_features
        rad_shs = rad_shs[sample_mask].transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)

        radiosity_result = rendering_equation_radiosity(
                            rad_base_color,
                            rad_roughness,
                            rad_normals,
                            rad_points,
                            rad_view,
                            pc, pipe=pipe, 
                            camera_center=viewpoint_camera.camera_center,
                            training=training,)
        
        if opt.rad_update_indirect:
            diffuse_incident_dirs = radiosity_result['incident_dirs'][:, :pipe.radiosity_sample_num]
            diffuse_incident_areas = torch.ones_like(diffuse_incident_dirs)[..., 0:1] * 2 * np.pi
            diffuse_visibility = radiosity_result['incident_visibility'][:, :pipe.radiosity_sample_num]
            diffuse_incident_lights = radiosity_result['local_incident_lights'][:, :pipe.radiosity_sample_num]
            pc.update_incidents_directions(diffuse_incident_dirs, diffuse_incident_areas, mask=sample_mask)
            pc.update_incident_radiance(diffuse_visibility, diffuse_incident_lights, detach=opt.rad_render_detach, mask=sample_mask)

        pbr_radiosity = radiosity_result['diffuse'] + radiosity_result['specular']
        nvs_radiosity = eval_sh(pc.active_sh_degree, rad_shs, -rad_view)
        nvs_radiosity = torch.clamp_min(nvs_radiosity + 0.5, 0.0)
        if True:
            NoV2 = (rad_normals * (rad_view)).sum(-1, keepdim=True)
            mask2 = torch.where(NoV2 > 0, 1.0, 0.0)
            pbr_radiosity = pbr_radiosity * mask2
            nvs_radiosity = nvs_radiosity * mask2

        if pipe.detach_rad_lhs: nvs_radiosity = nvs_radiosity.detach()
        if pipe.detach_rad_rhs: pbr_radiosity = pbr_radiosity.detach()
        
        results.update({
            "pbr_radiosity": pbr_radiosity,
            "nvs_radiosity": nvs_radiosity,
            "rad_f_d": radiosity_result["f_d"],
            "rad_f_s": radiosity_result["f_s"],
            "rad_ndf": radiosity_result["ndf"],
            "rad_energy": radiosity_result['energy'],
            "rad_roughness": rad_roughness.detach(),
        })

        if pipe.use_rad_rndview:
            # random view
            rand_rhs_list = []
            rand_lhs_list = []
            for _i in range(pipe.rndview_num):
                rad_view = random_hemisphere_sampling(rad_normals, sample_num=1)[:, 0]
                rad_view = rad_view / rad_view.norm(dim=-1, keepdim=True)
                rand_lhs_i = eval_sh(pc.active_sh_degree, rad_shs, -rad_view)
                rand_lhs_i = torch.clamp_min(rand_lhs_i + 0.5, 0.0)
                rand_radiosity = rendering_equation_radiosity(
                    rad_base_color,
                    rad_roughness,
                    rad_normals,
                    rad_points,
                    rad_view,
                    pc, pipe=pipe, 
                    precompute=True,
                    incident_dirs=radiosity_result['incident_dirs'],
                    incident_areas=radiosity_result['incident_areas'],
                    incident_visibility=radiosity_result['incident_visibility'],
                    local_incident_lights=radiosity_result['local_incident_lights'],
                    camera_center=viewpoint_camera.camera_center)
                rand_rhs_i = rand_radiosity['diffuse'] + rand_radiosity['specular']
                rand_rhs_list.append(rand_rhs_i)
                rand_lhs_list.append(rand_lhs_i)
            rand_rhs = torch.stack(rand_rhs_list, dim=0)
            rand_lhs = torch.stack(rand_lhs_list, dim=0)
            if pipe.detach_rad_lhs: rand_lhs = rand_lhs.detach()
            if pipe.detach_rad_rhs: rand_rhs = rand_rhs.detach()
            results.update({
                "rand_lhs": rand_lhs,
                "rand_rhs": rand_rhs,
            })

    
    # render per-Gaussian radiance
    if training:
        render_results = rendering_equation(base_color, roughness, normal, means3D, -dir_pp_normalized, pc, pipe=pipe, training=training, camera_center=viewpoint_camera.camera_center)
        diffuse = render_results['diffuse']
        specular = render_results['specular']
        light_direct = render_results['light_direct']
        pbr_features = torch.cat([diffuse, specular], dim=-1) # (N, 9)
    else:
        diffuse, specular, visibility, light_direct, light_indirect, direct_diffuse, direct_specular, indirect_diffuse, indirect_specular = [], [], [], [], [], [], [], [], []
        for i in range(0, base_color.shape[0], CHUNK_SIZE):
            render_results = rendering_equation(base_color[i:i+CHUNK_SIZE], roughness[i:i+CHUNK_SIZE], normal[i:i+CHUNK_SIZE], means3D[i:i+CHUNK_SIZE], -dir_pp_normalized[i:i+CHUNK_SIZE], pc, pipe=pipe, training=training, camera_center=viewpoint_camera.camera_center, chunk_idx=i)
            diffuse.append(render_results['diffuse'])
            specular.append(render_results['specular'])
            visibility.append(render_results['visibility'])
            light_direct.append(render_results['light_direct'])
            light_indirect.append(render_results['light_indirect'])
            direct_diffuse.append(render_results['direct_diffuse'])
            direct_specular.append(render_results['direct_specular'])
            indirect_diffuse.append(render_results['indirect_diffuse'])
            indirect_specular.append(render_results['indirect_specular'])
        diffuse = torch.cat(diffuse, 0)
        specular = torch.cat(specular, 0)
        visibility = torch.cat(visibility, 0)
        light_direct = torch.cat(light_direct, 0)
        light_indirect = torch.cat(light_indirect, 0)
        direct_diffuse = torch.cat(direct_diffuse, 0)
        direct_specular = torch.cat(direct_specular, 0)
        indirect_diffuse = torch.cat(indirect_diffuse, 0)
        indirect_specular = torch.cat(indirect_specular, 0)
        direct = direct_diffuse + direct_specular
        indirect = indirect_diffuse + indirect_specular
        torch.cuda.empty_cache()
        pbr_features = torch.cat([diffuse, specular, visibility, light_direct, light_indirect, light_direct+light_indirect, direct, indirect], dim=-1) # (N, 22)

        # for debugging, render per-Gaussian radiosity loss
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        nvs_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        radiosity = torch.abs(nvs_precomp - (diffuse + specular)).mean(dim=-1, keepdim=True)
        pbr_features = torch.cat([pbr_features, radiosity], dim=-1) # (N, 23)
    
    colors_precomp = diffuse+specular
    if pipe.bf_random:
        normal = normal / (normal.norm(dim=1, keepdim=True).clamp_min(1e-6))
        NoV = (normal * (-dir_pp_normalized)).sum(-1, keepdim=True)
        mask = torch.where(NoV > 0, 1.0, 0.0)
        colors_precomp = colors_precomp * mask
        if training and iteration > 5000:
            mask2 = (torch.rand_like(mask) < 0.3).float()
            colors_precomp = colors_precomp * mask + torch.rand_like(colors_precomp) * (1-mask) * mask2

    # rasterize second features
    contrib, pbr_rendered_image, pbr_rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        features = pbr_features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )

    if training:
        rendered_diffuse, rendered_specular = \
            torch.split(pbr_rendered_features, [3, 3], dim=0)
    else:
        rendered_diffuse, rendered_specular, rendered_visibility, \
            rendered_light_direct, rendered_light_indirect, rendered_light, \
            rendered_direct_full, rendered_indirect_full, rendered_radiosity = \
            torch.split(pbr_rendered_features, [3, 3, 1, 3, 3, 3, 3, 3, 1], dim=0)

    final_image = rgb_to_srgb(pbr_rendered_image) + bg_color[:, None, None] * (1 - render_alpha)        
    final_image_sh = rgb_to_srgb(rendered_image) + bg_color[:, None, None] * (1 - render_alpha)
    
    rays_d = viewpoint_camera.rays_d_hw
    direct_lights = rgb_to_srgb(pc.get_envmap(rays_d, mode='pure_env').permute(2,0,1))
    env_only = direct_lights

    if not training and bg_color.mean() > 0.0:
        rendered_roughness = rendered_roughness + bg_color[:, None, None] * (1 - render_alpha)
        rendered_base_color = rendered_base_color + bg_color[:, None, None] * (1 - render_alpha)
        rend_normal = render_normal + bg_color[:, None, None] * (1 - render_alpha)
        surf_normal = surf_normal + bg_color[:, None, None] * (1 - render_alpha)
    else: 
        rendered_roughness = rendered_roughness
        rendered_base_color = rendered_base_color
        rend_normal =  render_normal

    results.update({
        "render": final_image,
        "env_only": env_only,
        "render_sh": final_image_sh,
        "diffuse": rgb_to_srgb(rendered_diffuse),
        "specular": rgb_to_srgb(rendered_specular),
        "roughness": rendered_roughness,
        "base_color": rgb_to_srgb(rendered_base_color),
        "base_color_linear": rendered_base_color,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        'rend_alpha': render_alpha,
        'rend_normal': rend_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        "ray_light_direct": light_direct,
    })
    
    if not training:
        
        final_image_env = rgb_to_srgb(pbr_rendered_image) + direct_lights * (1 - render_alpha)
        rendered_light_direct = rendered_light_direct + bg_color[:, None, None] * (1 - render_alpha)
        rendered_light_indirect = rendered_light_indirect + bg_color[:, None, None] * (1 - render_alpha)
        rendered_direct_full = rendered_direct_full + bg_color[:, None, None] * (1 - render_alpha)
        rendered_indirect_full = rendered_indirect_full + bg_color[:, None, None] * (1 - render_alpha)
        

        results.update({
            "render_env": final_image_env,
            "light_direct": rgb_to_srgb(rendered_light_direct),
            "visibility": rendered_visibility,
            "light": rgb_to_srgb(rendered_light),
            "light_indirect": rgb_to_srgb(rendered_light_indirect),
            "render_radiosity": rendered_radiosity,
        })

        rendered_direct_diffuse = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_direct_specular = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_indirect_diffuse = torch.zeros_like(rendered_image).permute(1, 2, 0)
        rendered_indirect_specular = torch.zeros_like(rendered_image).permute(1, 2, 0)

        results.update({
            "render_direct": rgb_to_srgb(rendered_direct_full),
            "render_indirect": rgb_to_srgb(rendered_indirect_full),
        })

    return results

def rendering_equation_chunk(base_color, roughness, normal, position, w_o, pc, pipe, training=False, f0=0.02, relight=False, chunk_size=64, camera_center=None, image_sh=None, **kwargs):
    if (pipe.diffuse_sample_num + pipe.light_sample_num) <= chunk_size:
        return rendering_equation(base_color, roughness, normal, position, w_o, pc, pipe, training, f0, relight=relight, camera_center=camera_center, **kwargs)
    else:
        results = []
        for i in range(0, base_color.shape[0], chunk_size):
            results.append(rendering_equation(base_color[i:i+chunk_size], roughness[i:i+chunk_size], normal[i:i+chunk_size], position[i:i+chunk_size], w_o[i:i+chunk_size], pc, pipe, training, f0, relight=relight, camera_center=camera_center, **kwargs))
        return {k: torch.cat([r[k] for r in results], 0) for k in results[0]}
    
def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]

def rendering_equation(base_color, roughness, normals, position, viewdirs, pc, pipe, training=False, f0=0.04, relight=False, camera_center=None, **kwargs):
    B = base_color.shape[0]
    envmap = pc.get_envmap
    
    chunk_idx = kwargs.get('chunk_idx', None)
    incident_dirs = pc.get_incident_directions.clone()
    incident_areas = pc.get_incident_areas.clone()
    incident_visibility = pc.get_incident_visibility.clone()
    local_incident_lights = pc.get_local_incident_radiance.clone()
    if chunk_idx is not None:
        incident_dirs = incident_dirs[chunk_idx:chunk_idx+CHUNK_SIZE]
        incident_areas = incident_areas[chunk_idx:chunk_idx+CHUNK_SIZE]
        incident_visibility = incident_visibility[chunk_idx:chunk_idx+CHUNK_SIZE]
        local_incident_lights = local_incident_lights[chunk_idx:chunk_idx+CHUNK_SIZE]
    global_incident_lights = envmap(incident_dirs, mode='pure_env')    
    
    if relight:
        features = torch.cat([pc.get_base_color, pc.get_rough], dim=1)
        trace_outputs = pc.trace(position.unsqueeze(1)+incident_dirs*pipe.light_t_min, incident_dirs, features=features, camera_center=camera_center, back_culling=pipe.back_culling)
        trace_alpha = trace_outputs['alpha'][..., None]
        incident_visibility = 1 - trace_alpha
        trace_feature = trace_outputs['feature'] / trace_alpha.clamp_min(1e-6)
        trace_normal = F.normalize(trace_outputs['normal'], dim=-1)
        trace_base_color, trace_roughness = trace_feature.split([3, 1], dim=-1)
        trace_diffuse = trace_base_color * envmap(trace_normal, mode='diffuse')
        trace_wi = -incident_dirs
        trace_NdotV = (trace_normal * trace_wi).sum(-1, keepdim=True)
        trace_reflected = F.normalize(trace_NdotV * trace_normal * 2 - trace_wi, dim=-1)
        fg_uv = torch.cat([trace_NdotV, trace_roughness], -1).clamp(0, 1)
        fg = dr.texture(pc.FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(*fg_uv.shape)
        trace_specular = envmap(trace_reflected, roughness=trace_roughness, mode='specular') * (f0 * fg[..., 0:1] + fg[..., 1:2])
        local_incident_lights = (trace_diffuse + trace_specular) * trace_alpha
        if pipe.wo_indirect_relight:
            local_incident_lights = torch.zeros_like(local_incident_lights)
        incident_lights = incident_visibility * global_incident_lights + local_incident_lights
    else:
            
        if pipe.wo_indirect:
            local_incident_lights = torch.zeros_like(local_incident_lights)
        
    incident_lights = incident_visibility * global_incident_lights + local_incident_lights
    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] / np.pi
    f_s, ndf = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    direct_transport = incident_visibility * global_incident_lights * incident_areas * n_d_i  # (num_pts, num_sample, 3)
    indirect_transport = local_incident_lights * incident_areas * n_d_i  # (num_pts, num_sample, 3)
    diffuse = ((f_d) * transport).mean(dim=-2)
    specular = ((f_s) * transport).mean(dim=-2)
    direct_diffuse = ((f_d) * direct_transport).mean(dim=-2)
    direct_specular = ((f_s) * direct_transport).mean(dim=-2)
    indirect_diffuse = ((f_d) * indirect_transport).mean(dim=-2)
    indirect_specular = ((f_s) * indirect_transport).mean(dim=-2)

    if training:
        results = {
            "diffuse": diffuse,
            "specular": specular,
            "light_direct": global_incident_lights.mean(dim=1),
        }
    else:
        results = {
            "diffuse": diffuse,
            "specular": specular,
            "visibility": incident_visibility.mean(dim=1),
            "light": incident_lights.mean(dim=1),
            "light_indirect": local_incident_lights.mean(dim=1),
            "light_direct": global_incident_lights.mean(dim=1),
            "direct_diffuse": direct_diffuse,
            "direct_specular": direct_specular,
            "indirect_diffuse": indirect_diffuse,
            "indirect_specular": indirect_specular,
        }
    
    return results

def rendering_equation_radiosity(base_color, roughness, normals, position, viewdirs, pc, pipe, f0=0.04, camera_center=None, precompute=False, **kwargs):
    if not precompute:
        training=kwargs.get('training', False)
        with torch.no_grad():
            if pipe.use_rad_imp:
                light_sample_num = 0
                spec_sample_num = int(pipe.radiosity_sample_num * 0.04) + 1
                diff_sample_num = pipe.radiosity_sample_num
                
                incident_dirs, incident_areas = sample_mixture_directions(
                    normals, viewdirs, roughness, light_sample_num, diff_sample_num, spec_sample_num, pc, training=training and pipe.radiosity_random_sample)

            else: incident_dirs, incident_areas = sample_incident_rays(normals, pipe.radiosity_random_sample, pipe.radiosity_sample_num)
        trace_outputs = pc.trace(position.unsqueeze(1)+incident_dirs*pipe.light_t_min, incident_dirs, features=None, camera_center=camera_center, detach_orientation=pipe.detach_orientation, back_culling=pipe.back_culling)
        incident_visibility = 1 - trace_outputs['alpha'][..., None]
        local_incident_lights = trace_outputs['color']
    else:
        incident_dirs = kwargs['incident_dirs']
        incident_areas = kwargs['incident_areas']
        incident_visibility = kwargs['incident_visibility']
        local_incident_lights = kwargs['local_incident_lights']
        if pipe.use_rad_imp:
            light_sample_num = 0
            spec_sample_num = int(pipe.radiosity_sample_num * 0.04) + 1
            diff_sample_num = pipe.radiosity_sample_num
            assert incident_dirs.shape[1] == (diff_sample_num + spec_sample_num + light_sample_num), f"Expected {diff_sample_num + spec_sample_num + light_sample_num} incident directions, but got {incident_dirs.shape[1]}"
            diff_incident_dirs, diff_incident_areas = incident_dirs[:, :diff_sample_num, :], incident_areas[:,  :diff_sample_num, :]
            incident_dirs, incident_areas = add_specular_direction(normals, viewdirs, roughness, diff_sample_num, spec_sample_num, diff_incident_dirs, diff_incident_areas)
            spec_incident_dirs = incident_dirs[:, -spec_sample_num:, :]
            trace_spec_outputs = pc.trace(position.unsqueeze(1)+spec_incident_dirs*pipe.light_t_min, spec_incident_dirs, features=None, camera_center=camera_center, detach_orientation=pipe.detach_orientation, back_culling=pipe.back_culling)
            spec_incident_visibility = 1 - trace_spec_outputs['alpha'][..., None]
            spec_local_incident_lights = trace_spec_outputs['color']
            incident_visibility = torch.cat([incident_visibility[:, :diff_sample_num, :], spec_incident_visibility], dim=1)
            local_incident_lights = torch.cat([local_incident_lights[:, :diff_sample_num, :], spec_local_incident_lights], dim=1)

    global_incident_lights = pc.get_envmap(incident_dirs, mode='pure_env')
    if pipe.detach_rad_global: global_incident_lights = global_incident_lights.detach()
    if pipe.detach_rad_indirect: 
        local_incident_lights = local_incident_lights.detach()
        incident_visibility = incident_visibility.detach()
    incident_lights = incident_visibility * global_incident_lights + local_incident_lights
    f_s, ndf = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=f0)
    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] / np.pi
    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    diffuse = ((f_d) * transport).mean(dim=-2)
    specular = ((f_s) * transport).mean(dim=-2)
    results ={
        "diffuse": diffuse,
        "specular": specular,
        "f_d": f_d,
        "f_s": f_s,
        "ndf": ndf,
        "energy": ((f_d+f_s)*n_d_i*incident_areas).mean(1),
        "incident_dirs": incident_dirs,
        "incident_areas": incident_areas,
        "incident_visibility": incident_visibility,
        "local_incident_lights": local_incident_lights,
        "global_incident_lights": global_incident_lights,
    }
    return results

def GGX_specular(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    ndf = alpha2[:, None, :] / (np.pi * nom0 * nom0).clamp_(1e-6, np.pi)  # [nrays, nlights, 1]
    return spec, ndf

def sample_mixture_directions(normals, viewdirs,
                              roughness,
                              light_sample_num,
                              diff_sample_num,
                              spec_sample_num,
                              pc,
                              training=False):
    """
    Draws:
      - sample_num diffuse directions via sample_incident_rays
      - sample_num specular directions via GGX
      - pipe.light_sample_num env‐map directions
    Returns:
      dirs  : (B, sample_num*2 + L, 3)
      pdfs  : (B, sample_num*2 + L)
      weights: (B, sample_num*2 + L)  # for MIS (balance heuristic)
    """
    B = normals.shape[0]
    p_diffuse = diff_sample_num / (diff_sample_num + light_sample_num + spec_sample_num)
    p_light = light_sample_num / (diff_sample_num + light_sample_num + spec_sample_num)
    p_spec = spec_sample_num / (diff_sample_num + light_sample_num + spec_sample_num)

    incident_dirs_list, incident_pdfs_list = [], []

    # sample directions and pdfs
    if p_diffuse > 0:
        diffuse_directions, diffuse_areas = sample_incident_rays(normals, training, diff_sample_num)
        diffuse_pdfs = 1 / diffuse_areas
        if p_light > 0: light_pdfs_diffuse = pc.get_envmap.light_pdf(diffuse_directions)
        else: light_pdfs_diffuse = 0.0
        if p_spec > 0: spec_pdfs_diffuse = ggx_pdf(diffuse_directions, normals, viewdirs, roughness.squeeze(-1))
        else: spec_pdfs_diffuse = 0.0
        diffuse_pdfs = diffuse_pdfs * p_diffuse + spec_pdfs_diffuse * p_spec + light_pdfs_diffuse * p_light
        incident_dirs_list.append(diffuse_directions)
        incident_pdfs_list.append(diffuse_pdfs)                
    if p_light > 0: 
        light_directions, light_pdfs = pc.get_envmap.sample_light_directions(B,  light_sample_num, training)
        if p_diffuse > 0: diffuse_pdfs_light = 1 / (2 * np.pi)
        else: diffuse_pdfs_light = 0.0
        if p_spec > 0: spec_pdfs_light = ggx_pdf(light_directions, normals, viewdirs, roughness.squeeze(-1))
        else: spec_pdfs_light = 0.0
        light_pdfs = light_pdfs * p_light + diffuse_pdfs_light * p_diffuse + spec_pdfs_light * p_spec
        incident_dirs_list.append(light_directions)
        incident_pdfs_list.append(light_pdfs)
    if p_spec > 0: 
        # Calculate reflected direction (perfect reflection)
        dot = (normals * viewdirs).sum(-1, keepdim=True)  # [B, 1]
        reflected_dirs = viewdirs - 2 * dot * normals  # [B, 3]
        reflected_dirs = F.normalize(reflected_dirs, dim=-1)
        
        # For importance sampling, we'll use GGX distribution around the reflected direction
        # but ensure at least one sample is exactly the reflected direction
        spec_dirs_list = []
        spec_pdfs_list = []
        
        # First sample: exact reflected direction
        spec_dirs_list.append(reflected_dirs.unsqueeze(1))  # [B, 1, 3]
        
        # Calculate area for reflected direction using GGX PDF
        reflected_pdf = ggx_pdf(reflected_dirs.unsqueeze(1), normals, viewdirs, roughness.squeeze(-1))
        spec_pdfs_list.append(reflected_pdf.clamp_min(1e-6))  # [B, 1, 1]
        # Additional samples using GGX distribution if spec_sample_num > 1
        if spec_sample_num > 1:
            ggx_dirs, ggx_pdfs, _ = sample_ggx(normals, viewdirs, roughness.squeeze(-1), spec_sample_num - 1)
            spec_dirs_list.append(ggx_dirs)
            spec_pdfs_list.append(ggx_pdfs)
        specular_directions = torch.cat(spec_dirs_list, dim=1)  # [B, spec_sample_num, 3]
        specular_pdfs = torch.cat(spec_pdfs_list, dim=1)  #
        # specular_directions, specular_pdfs, F_spec = sample_ggx(normals, viewdirs, roughness.squeeze(-1), spec_sample_num)
        if p_diffuse > 0: diffuse_pdfs_spec = 1/  (2 * np.pi)
        else: diffuse_pdfs_spec = 0.0
        if p_light > 0: light_pdfs_spec = pc.get_envmap.light_pdf(specular_directions)
        else: light_pdfs_spec = 0.0
        specular_pdfs = specular_pdfs * p_spec + diffuse_pdfs_spec * p_diffuse + light_pdfs_spec * p_light
        incident_dirs_list.append(specular_directions)
        incident_pdfs_list.append(specular_pdfs)
    
    incident_dirs = torch.cat(incident_dirs_list, dim=1)
    incident_pdfs = torch.cat(incident_pdfs_list, dim=1)
    incident_areas = 1 / incident_pdfs.clamp_min(1e-6)

    return incident_dirs, incident_areas


def add_specular_direction(normals, viewdirs, roughness, diff_sample_num, spec_sample_num, diff_incident_dirs, diff_incident_areas):
    B, S, _ = diff_incident_dirs.shape
    if spec_sample_num == 0:
        return diff_incident_areas, None
    else:
        # Calculate reflected direction (perfect reflection)
        dot = (normals * viewdirs).sum(-1, keepdim=True)  # [B, 1]
        reflected_dirs = viewdirs - 2 * dot * normals  # [B, 3]
        reflected_dirs = F.normalize(reflected_dirs, dim=-1)
        
        # For importance sampling, we'll use GGX distribution around the reflected direction
        # but ensure at least one sample is exactly the reflected direction
        spec_dirs_list = []
        spec_pdfs_list = []
        
        # First sample: exact reflected direction
        spec_dirs_list.append(reflected_dirs.unsqueeze(1))  # [B, 1, 3]
        
        # Calculate area for reflected direction using GGX PDF
        reflected_pdf = ggx_pdf(reflected_dirs.unsqueeze(1), normals, viewdirs, roughness.squeeze(-1))
        spec_pdfs_list.append(reflected_pdf.clamp_min(1e-6))  # [B, 1, 1]
        # Additional samples using GGX distribution if spec_sample_num > 1
        if spec_sample_num > 1:
            ggx_dirs, ggx_pdfs, _ = sample_ggx(normals, viewdirs, roughness.squeeze(-1), spec_sample_num - 1)
            spec_dirs_list.append(ggx_dirs)
            spec_pdfs_list.append(ggx_pdfs)
        spec_incident_dirs = torch.cat(spec_dirs_list, dim=1)  # [B, spec_sample_num, 3]
        spec_pdfs = torch.cat(spec_pdfs_list, dim=1)  # [B, spec_sample_num, 1]

        # combine pdfs using MIS (balance heuristic)
        p_spec = spec_sample_num / (diff_sample_num + spec_sample_num)
        p_diffuse = diff_sample_num / (diff_sample_num + spec_sample_num)
        diff_pdfs = 1 / diff_incident_areas.clamp_min(1e-6)
        spec_pdfs_diffuse = ggx_pdf(diff_incident_dirs, normals, viewdirs, roughness.squeeze(-1))
        diff_pdf_specular = 1 / (2 * np.pi)
        spec_pdfs = spec_pdfs * p_spec + diff_pdf_specular * p_diffuse
        diff_pdfs = diff_pdfs * p_diffuse + spec_pdfs_diffuse * p_spec
        incident_dirs = torch.cat([diff_incident_dirs, spec_incident_dirs], dim=1)
        incident_areas = torch.cat([1 / diff_pdfs.clamp_min(1e-6), 1 / spec_pdfs.clamp_min(1e-6)], dim=1)
        return incident_dirs, incident_areas