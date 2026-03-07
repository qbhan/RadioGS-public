import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from scene.light import EnvLight
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, rgb_to_srgb
from utils.general_utils import strip_symmetric, build_scaling_rotation, safe_normalize, flip_align_view, rotation_to_quaternion, quaternion_multiply
from surfel_tracer import GaussianTracer
import trimesh
from utils.system_utils import Timing

def get_env_direction1(H, W):
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 0.5 / H, 1.0 - 0.5 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            indexing='ij')
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    env_directions = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return env_directions


def get_env_direction2(H, W):
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 0.5 / H, 1.0 - 0.5 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            indexing='ij')
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    env_directions = torch.stack((
        sintheta*cosphi,
        -sintheta*sinphi, 
        costheta, 
        ), dim=-1)
    return env_directions


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.base_color_activation = lambda x: torch.sigmoid(x) * 0.77 + 0.03
        self.inverse_base_color_activation = lambda x: inverse_sigmoid(x-0.03) / 0.77
        
        self.metallic_ativation = torch.sigmoid
        self.inverse_metallic_activation = inverse_sigmoid

        self.roughness_activation = torch.sigmoid
        self.inverse_roughness_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._base_color = torch.empty(0) 
        self._metallic = torch.empty(0) 
        self._roughness = torch.empty(0) 
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.init_roughness_value = 0.5
        self.init_metallic_value = 0.5
        self.init_base_color_value = 0.5
        self.enlarge_scale = 1.5
        self.metallic_msk_thr = 0.02
        self.rough_msk_thr = 0.2

        self.env_map = None
        self.env_H, self.env_W = 256, 512
        self.env_directions1 = get_env_direction1(self.env_H, self.env_W)
        self.env_directions2 = get_env_direction2(self.env_H, self.env_W)
        self.ray_tracer = None
        self.setup_functions()
        
        
        # icosahedron, outer sphere radius is 1.0
        icosahedron = trimesh.creation.icosahedron()
        
        # change to inner sphere radius equal to 1.0
        # the central point of each face must be on the unit sphere
        self.unit_icosahedron_vertices = torch.from_numpy(icosahedron.vertices).float().cuda() * 1.2584 
        self.unit_icosahedron_faces = torch.from_numpy(icosahedron.faces).long().cuda()
        
        self.gaussian_tracer = GaussianTracer(transmittance_min=0.03)
        self.alpha_min = 1 / 255
        
        self.FG_LUT = torch.from_numpy(np.fromfile("assets/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2)).cuda()
        
    @torch.no_grad()
    def set_transform(self, rotation=None, center=None, scale=None, offset=None, transform=None):
        if transform is not None:
            scale = transform[:3, :3].norm(dim=-1)
            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale[:2])
            xyz_homo = torch.cat([self._xyz.data, torch.ones_like(self._xyz[:, :1])], dim=-1)
            self._xyz.data = (xyz_homo @ transform.T)[:, :3]
            rotation = transform[:3, :3] / scale[:, None]
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
            return

        if center is not None:
            self._xyz.data = self._xyz.data - center
        if rotation is not None:
            self._xyz.data = (self._xyz.data @ rotation.T)
            self._normal.data = self._normal.data @ rotation.T
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
        if scale is not None:
            self._xyz.data = self._xyz.data * scale
            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale)
        if offset is not None:
            self._xyz.data = self._xyz.data + offset
            
    @property
    def attribute_names(self):
        attribute_names = ['xyz', 'base_color', 'metallic', 'roughness', 'features_dc', 'features_rest', 'scaling', 
                           'rotation', 'opacity']
        return attribute_names

    @classmethod
    def create_from_gaussians(cls, gaussians_list, dataset):
        assert len(gaussians_list) > 0
        gaussians = GaussianModel(sh_degree=3)
        attribute_names = gaussians.attribute_names
        for attribute_name in attribute_names:
            setattr(gaussians, "_" + attribute_name,
                    nn.Parameter(torch.cat([getattr(g, "_" + attribute_name).data.cuda() for g in gaussians_list],
                                           dim=0).requires_grad_(True)))

        return gaussians

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._metallic, 
            self._roughness, 
            self._base_color, 
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.env_map.capture(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args=None):
        (self.active_sh_degree, 
        self._xyz, 
        self._metallic, 
        self._roughness, 
        self._base_color, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        env_dict,
        self.spatial_lr_scale) = model_args
        self.env_map.restore(env_dict)
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    def restore_from_refgs(self, model_args, training_args=None):
        
        if len(model_args) == 26:
            (self.active_sh_degree, 
            self._xyz, 
            _refl_strength,  
            self._metallic, 
            self._roughness, 
            self._base_color, 
            _diffuse_color,
            self._features_dc, 
            self._features_rest,
            _indirect_dc, 
            _indirect_rest,
            _indirect_asg,
            _visibility_dc, 
            _visibility_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self._normal1,  
            self._normal2,  
            self.max_radii2D, 
            xyz_gradient_accum, 
            normal_gradient_accum, 
            denom,
            opt_dict, 
            env_dict,
            self.spatial_lr_scale) = model_args
        else:
            if len(model_args) == 19:
                (self.active_sh_degree, 
                self._xyz, 
                self._metallic, 
                self._roughness, 
                self._base_color, 
                self._features_dc, 
                self._features_rest,
                _indirect_dc, 
                _indirect_rest,
                self._scaling, 
                self._rotation, 
                self._opacity,
                self.max_radii2D, 
                xyz_gradient_accum, 
                denom,
                opt_dict, 
                env_1_dict,
                env_2_dict,
                self.spatial_lr_scale) = model_args
            elif len(model_args) == 20:
                (self.active_sh_degree, 
                self._xyz, 
                self._metallic, 
                self._roughness, 
                self._base_color, 
                self._features_dc, 
                self._features_rest,
                _indirect_dc, 
                _indirect_rest,
                self._scaling, 
                self._rotation, 
                self._opacity,
                self.max_radii2D, 
                xyz_gradient_accum,
                radiosity_accum, 
                denom,
                opt_dict, 
                env_1_dict,
                env_2_dict,
                self.spatial_lr_scale) = model_args
            
        self._base_color.data[:] = self.inverse_base_color_activation(torch.full_like(self._base_color.data, self.init_base_color_value))
        self._metallic.data[:] = self.inverse_metallic_activation(torch.full_like(self._metallic.data, self.init_metallic_value))
        self._roughness.data[:] = self.inverse_roughness_activation(torch.full_like(self._roughness.data, self.init_roughness_value))

        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            
    def set_opacity_lr(self, lr):   
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "opacity":
                param_group['lr'] = lr

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) 
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_metallic(self): 
        return self.metallic_ativation(self._metallic)

    @property
    def get_rough(self): 
        return self.roughness_activation(self._roughness)

    @property
    def get_base_color(self): 
        return self.base_color_activation(self._base_color)
    
    def get_normal(self, scaling_modifier, dir_pp_normalized=None): 
        splat2world = self.get_covariance(scaling_modifier)
        normals_raw = splat2world[:,2,:3] 
        if dir_pp_normalized is not None:
            normals_raw, positive = flip_align_view(normals_raw, dir_pp_normalized)
        normals = safe_normalize(normals_raw)
        return normals

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    def render_env_map(self, H=512):
        if H == self.env_H:
            directions1 = self.env_directions1
            directions2 = self.env_directions2
        else:
            W = H * 2
            directions1 = get_env_direction1(H, W)
            directions2 = get_env_direction2(H, W)
        return {'env1': self.env_map(directions1, mode="pure_env"), 'env2': self.env_map(directions2, mode="pure_env")}
    
    @property   
    def get_envmap(self): 
        return self.env_map
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, args):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        sh_features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        sh_features[:, :3, 0 ] = fused_color
        sh_features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        base_color = self.inverse_base_color_activation(torch.full_like(fused_point_cloud, self.init_base_color_value))
        metallic = self.inverse_metallic_activation(torch.full_like(opacities, self.init_metallic_value))
        roughness = self.inverse_roughness_activation(torch.full_like(opacities,  self.init_roughness_value))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._base_color = nn.Parameter(base_color.requires_grad_(True)) 
        self._roughness = nn.Parameter(roughness.requires_grad_(True)) 
        self._metallic = nn.Parameter(metallic.requires_grad_(True)) 
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._features_dc = nn.Parameter(sh_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(sh_features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        self.env_map = EnvLight(path=None, device='cuda', resolution=[args.envmap_resolution // 2, args.envmap_resolution], max_res=args.envmap_resolution, init_value=args.envmap_init_value, activation=args.envmap_activation).cuda()
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.features_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.features_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.env_map.parameters(), 'lr': training_args.envmap_cubemap_lr, "name": "env"},     
        ]

        l.extend([
            {'params': [self._base_color], 'lr': training_args.base_color_lr, "name": "base_color"},  
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},  
            {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},  
        ])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('metallic') 
        l.append('roughness') 
        for i in range(self._base_color.shape[1]):
            l.append('base_color_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        metallic = self._metallic.detach().cpu().numpy()    
        roughness = self._roughness.detach().cpu().numpy()    
        base_color = self._base_color.detach().cpu().numpy()    
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, f_rest, opacities, metallic, roughness, base_color, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        if self.env_map is not None:
            save_path = path.replace('.ply', '1.map')
            torch.save(self.env_map.capture(), save_path)

    def reset_opacity_mask0(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_mask1(self, exclusive_msk = None):
        RESET_V = 0.9
        opacity_old = self.get_opacity
        o_msk = (opacity_old > RESET_V).flatten()
        if exclusive_msk is not None:
            o_msk = torch.logical_or(o_msk, exclusive_msk)
        opacities_new = torch.ones_like(opacity_old)*inverse_sigmoid(torch.tensor([RESET_V]).cuda())
        opacities_new[o_msk] = self._opacity[o_msk]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if "opacity" not in optimizable_tensors: return
        self._opacity = optimizable_tensors["opacity"]

    def enlarge_scales(self, ret_raw=True, ENLARGE_SCALE=1.5, METALLIC_MSK_THR=0.02, ROUGH_MSK_THR=0.2, exclusive_msk=None):
        ENLARGE_SCALE = self.enlarge_scale
        METALLIC_MSK_THR = self.metallic_msk_thr
        ROUGH_MSK_THR = self.rough_msk_thr

        rough_msk = self.get_rough.flatten() > ROUGH_MSK_THR
        combined_msk = rough_msk
        if exclusive_msk is not None:
            combined_msk = torch.logical_or(combined_msk, exclusive_msk) 
        scales = self.get_scaling
        rmin_axis = (torch.ones_like(scales) * ENLARGE_SCALE)
        if ret_raw:
            scale_new = self.scaling_inverse_activation(scales * rmin_axis)
            scale_new[combined_msk] = self._scaling[combined_msk]
        else:
            scale_new = scales * rmin_axis
            scale_new[combined_msk] = scales[combined_msk]   
        return scale_new

    def reset_scale(self, exclusive_msk = None):
        scale_new = self.enlarge_scales(ret_raw=True, exclusive_msk=exclusive_msk)
        optimizable_tensors = self.replace_tensor_to_optimizer(scale_new, "scaling")
        if "scaling" not in optimizable_tensors: return
        self._scaling = optimizable_tensors["scaling"]

    def load_ply(self, path, **kwargs):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        base_color = np.stack((np.asarray(plydata.elements[0]['base_color_0']),
                              np.asarray(plydata.elements[0]['base_color_1']),
                              np.asarray(plydata.elements[0]['base_color_2'])),  axis=1)
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis] # #
        metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis] # #
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        self.active_sh_degree = self.max_sh_degree
        
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        map_path = path.replace('.ply', '1.map')
        if os.path.exists(map_path):
            map_ckpt = torch.load(map_path)
            self.env_map = EnvLight(path=None, device='cuda', resolution=map_ckpt['state_dict']['base'].shape[:2]).cuda()
            self.env_map.restore(map_ckpt)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._base_color = nn.Parameter(torch.tensor(base_color, dtype=torch.float, device="cuda").requires_grad_(True))   # #
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None: continue
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "env": continue   # #
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]

        self._base_color = optimizable_tensors['base_color']    # #
        self._roughness = optimizable_tensors['roughness']    # #
        self._metallic = optimizable_tensors['metallic']    # #

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "env": continue   # #
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_metallic, new_roughness, new_base_color, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "metallic": new_metallic, 
            "roughness": new_roughness,
            "base_color": new_base_color,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]

        self._metallic = optimizable_tensors['metallic']
        self._roughness = optimizable_tensors['roughness']
        self._base_color = optimizable_tensors['base_color']
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_base_color = self._base_color[selected_pts_mask].repeat(N,1)   # #
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)   # #
        new_metallic = self._metallic[selected_pts_mask].repeat(N,1)   # #

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_metallic, new_roughness, new_base_color, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]

        new_metallic = self._metallic[selected_pts_mask]   # #
        new_roughness = self._roughness[selected_pts_mask]   # #
        new_base_color = self._base_color[selected_pts_mask]   # #

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_metallic, new_roughness, new_base_color, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)  # #
        self.denom[update_filter] += 1

    def get_boundings(self, alpha_min=0.01):
        mu = self.get_xyz
        opacity = self.get_opacity
        scale = self.get_scaling
        scale = torch.cat([scale, torch.full_like(scale, 1e-6)], dim=-1)
        
        L = build_scaling_rotation(scale, self._rotation)
        
        vertices_b = (2 * (opacity/alpha_min).log()).sqrt()[:, None] * (self.unit_icosahedron_vertices[None] @ L.transpose(-1, -2)) + mu[:, None]
        faces_b = self.unit_icosahedron_faces[None] + torch.arange(mu.shape[0], device="cuda")[:, None, None] * 12
        gs_id = torch.arange(mu.shape[0], device="cuda")[:, None].expand(-1, faces_b.shape[1])
        return vertices_b.reshape(-1, 3), faces_b.reshape(-1, 3), gs_id.reshape(-1)
    
    def build_bvh(self):
        vertices_b, faces_b, gs_id = self.get_boundings(alpha_min=self.alpha_min)
        self.gaussian_tracer.build_bvh(vertices_b, faces_b, gs_id)
        
    def update_bvh(self):
        vertices_b, faces_b, gs_id = self.get_boundings(alpha_min=self.alpha_min)
        self.gaussian_tracer.update_bvh(vertices_b, faces_b, gs_id)
        
    def trace(self, rays_o, rays_d, features=None, camera_center=None, back_culling=False, detach_orientation=False):
        means3D = self.get_xyz
        shs = self.get_features
        opacity = self.get_opacity
        
        s = 1 / self.get_scaling
        R = build_rotation(self._rotation)
        ru = R[:, :, 0] * s[:,0:1]
        rv = R[:, :, 1] * s[:,1:2]
        
        splat2world = self.get_covariance()
        normals_raw = splat2world[: ,2, :3] 
        if camera_center is not None:
            normals_raw, positive = flip_align_view(normals_raw, means3D - camera_center)
        normals = safe_normalize(normals_raw)
        
        if not detach_orientation: color, normal, feature, depth, alpha = self.gaussian_tracer.trace(rays_o, rays_d, means3D, opacity, ru, rv, normals, features, shs, alpha_min=self.alpha_min, deg=self.active_sh_degree, back_culling=back_culling)
        else: color, normal, feature, depth, alpha = self.gaussian_tracer.trace(rays_o.detach(), rays_d.detach(), means3D.detach(), opacity.detach(), ru.detach(), rv.detach(), normals.detach(), features, shs, alpha_min=self.alpha_min, deg=self.active_sh_degree, back_culling=back_culling)
        
        alpha_ = alpha[..., None]
        color = torch.where(alpha_ < 1 - self.gaussian_tracer.transmittance_min, color, color / alpha_)
        normal = torch.where(alpha_ < 1 - self.gaussian_tracer.transmittance_min, normal, normal / alpha_)
        feature = torch.where(alpha_ < 1 - self.gaussian_tracer.transmittance_min, feature, feature / alpha_)
        depth = torch.where(alpha < 1 - self.gaussian_tracer.transmittance_min, depth, depth / alpha)
        alpha = torch.where(alpha < 1 - self.gaussian_tracer.transmittance_min, alpha, torch.ones_like(alpha))
        
        return {
            "color": color,
            "normal": normal,
            "feature": feature,
            "depth": depth,
            "alpha" : alpha,
            "normals": normals,
        }
