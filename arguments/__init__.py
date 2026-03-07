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

from argparse import ArgumentParser, Namespace
import sys
import os
from . import refgs

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs="+")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, default=value, nargs="+")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        # Rendering Settings
        self.sh_degree = 3
        self._resolution = -1
        self._white_background = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        self.batch_size = 2**16
        self.transmittance_min = 0.03
        
        # Paths
        self._source_path = ""
        self._model_path = ""
        self._images = "images"

        # Device Settings
        self.data_device = "cuda"
        self.eval = False

        # EnvLight Settings
        self.envmap_resolution = 8
        self.relight = False
        self.envmap_init_value = 1.5
        self.envmap_activation = 'exp'

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        # training resume setting
        self.restart = False

        # Processing Settings
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        
        # tone mapping settings
        self.tone_map = False
        self.no_gamma = False

        # Debugging
        self.depth_ratio = 0.0
        self.debug = False
        self.light_sample_num = 0
        self.diffuse_sample_num = 256
        self.specular_sample_num = 0
        self.light_t_min = 0.05
        
        self.wo_indirect = False
        self.wo_indirect_relight = False
        self.detach_indirect = False

        # for ablation on ray tracing gradients
        self.detach_orientation = False
        self.detach_orientation_raster = False

        # Radiosity Settings
        self.use_radiosity = False
        self.radiosity_gaussian_num = 0
        self.radiosity_sample_num = 0
        self.detach_rad_mat = False
        self.detach_rad_normal = False
        self.detach_rad_global = False
        self.detach_rad_indirect = False
        self.use_rad_imp = False
        self.use_rad_rndview = False
        self.rndview_num = 1
        self.detach_rad_lhs = False
        self.detach_rad_rhs = False
        self.back_culling = False
        self.bf_random = False
        self.radiosity_random_sample = 1

        # ablation on view
        self.view_ratio = 1.0

        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Learning Rate Settings
        self.iterations = 60_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.features_lr = 0.0075 
        self.indirect_lr = 0.0075 
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.lr_scale = 0.0

        self.base_color_lr = 0.0075 
        self.metallic_lr =  0.0
        self.roughness_lr =  0.005 
        self.normal_lr = 0.006
        self.envmap_cubemap_lr = 0.1
        
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_pbr = 1.0
        self.lambda_nvs = 1.0
        self.lambda_normal_render_depth = 0.05
        self.lambda_normal_smooth = 0.01
        self.lambda_depth_smooth = 0.0
        self.lambda_mask_entropy = 0.01
        
        self.lambda_base_color_smooth = 0.0
        self.lambda_roughness_smooth = 0.0
        self.lambda_metallic_smooth = 0.0
        self.lambda_light = 0.0
        self.lambda_light_smooth = 0.0

        self.init_roughness_value = 0.7
        self.init_base_color_value = 0.3
        self.init_metallic_value = 0.01

        self.percent_dense = 0.01
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 10000 
        self.densify_grad_threshold = 0.0002
        self.prune_opacity_threshold = 0.005
        self.indirect_update_interval = 100
        self.only_rnd_view = False

        self.normal_loss_start = 1000
        self.dist_loss_start = 1000
        self.radiosity_loss_start = 0
        self.lambda_radiosity = 0.0
        self.rad_loss = 'l1'  # Options: l1, l2, relmse, smape
        self.weight_roughness = False
        self.rad_render_detach = 1
        self.rad_update_indirect = 1
        self.pbr_loss_start = 0
        self.rad_only_rndview = False

        # dummy for debug compatibility
        self.train_sh_vol = False
        self.train_sh_surf = False
        self.train_sh_surf_from_iter = 0
        self.lambda_radiosity = 0.0
        self.radiosity = False
        self.radiosity_from_iter = 50000
        self.normal_smooth_from_iter = 0
        self.normal_smooth_until_iter = 18000
        self.init_until_iter = 2000
        self.normal_prop_until_iter = 10_000 
        self.normal_prop_interval = 1000
        self.opac_lr0_interval = 200
        self.densification_interval_when_prop = 500
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
