import json
import os
from gaussian_renderer import render_radiogs
import numpy as np
import torch
import torch.nn.functional as F
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
from scene.dataset_readers import load_img_rgb
from os import makedirs
import torchvision
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


def get_mae(gt_normal, normal):
    mae = (gt_normal*normal).sum(0).clamp(-1, 1).arccos().mean() * 180 / np.pi
    return mae
    
if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--albedo_rescale", default=2, type=int, help="0: no scale; 1: single channel scale; 2: three channel scale")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--no_save", default=False, action='store_true')
    parser.add_argument("--no_lpips", default=False, action='store_true')
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    # load gaussians
    # 
    gaussians = RadioGSModel(3)
    
    if args.iteration < 0:
        loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    else:
        loaded_iter = args.iteration
    gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply")) 
    gaussians.build_bvh()
        
    # deal with each item
    test_transforms_file = os.path.join(args.source_path, "transforms_test.json")
    contents = load_json_config(test_transforms_file)

    fovx = contents["camera_angle_x"]
    frames = contents["frames"]

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    if args.albedo_rescale == 0:
        base_color_scale = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    else:
        with open(os.path.join(args.model_path, "albedo_scale.json"), "r") as f:
            albedo_scale_dict = json.load(f)
        base_color_scale = torch.tensor(albedo_scale_dict[str(args.albedo_rescale)], dtype=torch.float32, device="cuda")
    
    render_kwargs = {
        "pc": gaussians,
        "pipe": pipe,
        "bg_color": background,
        "training": False,
        "relight": False,
        "base_color_scale": base_color_scale,
        "material_only": True,
    }
    
    psnr_albedo_linear = 0.0
    ssim_albedo_linear = 0.0
    lpips_albedo_linear = 0.0
    psnr_albedo = 0.0
    ssim_albedo = 0.0
    lpips_albedo = 0.0
    mae_normal = 0.0
    results_dict = {}
    normal_bg = torch.tensor([0.0, 0.0, 1.0], device='cuda')
    
    # <<< CHANGE START >>>
    render_times = []
    pbar = tqdm(frames, desc="Rendering and Evaluating Materials")
    # <<< CHANGE END >>>

    for idx, frame in enumerate(pbar):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        albedo_path = os.path.join(args.source_path, frame["file_path"].replace("rgba", "albedo.png"))
        gt_albedo_np = load_img_rgb(albedo_path)
        gt_albedo = torch.from_numpy(gt_albedo_np[..., :3] * gt_albedo_np[..., 3:4]).permute(2, 0, 1).float().cuda()
        mask = torch.from_numpy(gt_albedo_np[..., 3:]).permute(2, 0, 1).float().cuda()
        gt_albedo_linear = gt_albedo
        gt_albedo = rgb_to_srgb(gt_albedo)
        
        H = gt_albedo.shape[1]
        W = gt_albedo.shape[2]
        fovy = focal2fov(fov2focal(fovx, W), H)

        custom_cam = Camera(colmap_id=0, R=R, T=T,
                            FoVx=fovx, FoVy=fovy,
                            image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name=None, uid=0)

        with torch.no_grad():
            # <<< CHANGE START >>>
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            render_pkg = render_radiogs(viewpoint_camera=custom_cam, **render_kwargs)

            end_time.record()
            torch.cuda.synchronize()
            render_time = start_time.elapsed_time(end_time)
            render_times.append(render_time)

            pbar.set_postfix({
                'Time': f'{render_time:.1f}ms',
                'FPS': f'{1000.0 / render_time:.1f}',
                'Avg': f'{np.mean(render_times):.1f}ms'
            })
            # <<< CHANGE END >>>

        makedirs(os.path.join(args.model_path, "material_results"), exist_ok=True)
        makedirs(os.path.join(args.model_path, "material_results", "base_color_linear"), exist_ok=True)
        makedirs(os.path.join(args.model_path, "material_results", "base_color"), exist_ok=True)
        makedirs(os.path.join(args.model_path, "material_results", "base_color_linear_white"), exist_ok=True)
        makedirs(os.path.join(args.model_path, "material_results", "base_color_white"), exist_ok=True)
        makedirs(os.path.join(args.model_path, "material_results", "base_color_err_map"), exist_ok=True)
        makedirs(os.path.join(args.model_path, "material_results", "base_color_err_map05"), exist_ok=True)
        makedirs(os.path.join(args.model_path, "material_results", "base_color_err_map_inferno"), exist_ok=True)
        makedirs(os.path.join(args.model_path, "material_results", "base_color_err_map_inferno05"), exist_ok=True)
        makedirs(os.path.join(args.model_path, "material_results", "gt_albedo"), exist_ok=True)

        base_color_linear_white = render_pkg['base_color_linear'] + (1.0 - render_pkg['rend_alpha'])
        base_color_white = render_pkg['base_color'] + (1.0 - render_pkg['rend_alpha'])
        if not args.no_save: torchvision.utils.save_image(base_color_linear_white, os.path.join(args.model_path, "material_results", "base_color_linear_white", '{0:05d}'.format(idx) + ".png"))
        if not args.no_save: torchvision.utils.save_image(base_color_white, os.path.join(args.model_path, "material_results", "base_color_white", '{0:05d}'.format(idx) + ".png"))

        gt_albedo_white = gt_albedo + (1.0 - mask)
        if not args.no_save: torchvision.utils.save_image(gt_albedo_white, os.path.join(args.model_path, "material_results", "gt_albedo", '{0:05d}'.format(idx) + ".png"))

        render_pkg['base_color_linear'] = render_pkg['base_color_linear'] * mask
        render_pkg['roughness'] = render_pkg['roughness'] * mask
        render_pkg['base_color'] = render_pkg['base_color'] * mask
        gt_albedo_linear = gt_albedo_linear * mask
        gt_albedo = gt_albedo * mask

        render_pkg["base_color"] = render_pkg["base_color"] * mask
        render_pkg["base_color_linear"] = render_pkg["base_color_linear"] * mask
        base_color_err_map = torch.abs(render_pkg['base_color'] - gt_albedo).mean(dim=0, keepdim=True)
        if not args.no_save: torchvision.utils.save_image(render_pkg['base_color_linear'], os.path.join(args.model_path, "material_results", "base_color_linear", '{0:05d}'.format(idx) + ".png"))
        if not args.no_save: torchvision.utils.save_image(render_pkg['base_color'], os.path.join(args.model_path, "material_results", "base_color", '{0:05d}'.format(idx) + ".png"))
        if not args.no_save: torchvision.utils.save_image(base_color_err_map, os.path.join(args.model_path, "material_results", "base_color_err_map", '{0:05d}'.format(idx) + ".png"))
        cmap = plt.get_cmap('inferno')
        colored_err_map = cmap(base_color_err_map[0].cpu().numpy())
        colored_err_map = torch.from_numpy(colored_err_map).permute(2, 0, 1).float().cuda()
        if not args.no_save: save_image(colored_err_map, os.path.join(args.model_path, "material_results", "base_color_err_map_inferno", '{0:05d}'.format(idx) + ".png"))
        if not args.no_save: torchvision.utils.save_image(base_color_err_map*2, os.path.join(args.model_path, "material_results", "base_color_err_map05", '{0:05d}'.format(idx) + ".png"))
        if not args.no_save: save_image(colored_err_map*2, os.path.join(args.model_path, "material_results", "base_color_err_map_inferno05", '{0:05d}'.format(idx) + ".png"))
        psnr_albedo_linear += psnr(render_pkg['base_color_linear'], gt_albedo_linear).mean().double().item()
        psnr_albedo += psnr(render_pkg['base_color'], gt_albedo).mean().double().item()
        ssim_albedo_linear += ssim(render_pkg['base_color_linear'], gt_albedo_linear).mean().double().item()
        ssim_albedo += ssim(render_pkg['base_color'], gt_albedo).mean().double().item()
        if not args.no_lpips:
            lpips_albedo_linear += lpips(render_pkg['base_color_linear'], gt_albedo_linear, net_type='vgg').mean().double().item()
            lpips_albedo += lpips(render_pkg['base_color'], gt_albedo, net_type='vgg').mean().double().item()
        
        normal = render_pkg['rend_normal']
        alpha = render_pkg['rend_alpha']
        normal = normal * alpha + normal_bg[:, None, None] * (1.0 - alpha)
        normal = F.normalize(normal, dim=0)
        
        normal_path = os.path.join(args.source_path, frame["file_path"].replace("rgba", "normal.png"))
        gt_normal_img = torch.from_numpy(load_img_rgb(normal_path)[..., :3]).float().cuda().permute(2, 0, 1)
        gt_normal = (gt_normal_img - 0.5) * 2.0
        gt_normal = gt_normal * mask + normal_bg[:, None, None] * (1.0 - mask)
        gt_normal = F.normalize(gt_normal, dim=0)
        mae_normal += get_mae(gt_normal, normal).mean().double().item()
                
    psnr_albedo_linear /= len(frames)
    ssim_albedo_linear /= len(frames)
    lpips_albedo_linear /= len(frames)
    psnr_albedo /= len(frames)
    ssim_albedo /= len(frames)
    lpips_albedo /= len(frames)
    mae_normal /= len(frames)

    # <<< CHANGE START >>>
    avg_render_time_ms = np.mean(render_times)
    total_render_time_ms = np.sum(render_times)
    fps = 1000.0 / avg_render_time_ms
    # <<< CHANGE END >>>
        
    results_dict["psnr_albedo_linear_avg"] = psnr_albedo_linear
    results_dict["ssim_albedo_linear_avg"] = ssim_albedo_linear
    results_dict["lpips_albedo_linear_avg"] = lpips_albedo_linear
    results_dict["psnr_albedo_avg"] = psnr_albedo
    results_dict["ssim_albedo_avg"] = ssim_albedo
    results_dict["lpips_albedo_avg"] = lpips_albedo
    results_dict["mae_normal_avg"] = mae_normal
    
    # <<< CHANGE START >>>
    results_dict["avg_render_time_ms"] = avg_render_time_ms
    results_dict["total_render_time_ms"] = total_render_time_ms
    results_dict["fps"] = fps
    # <<< CHANGE END >>>
    
    print("\nEvaluating AVG: PSNR_ALBEDO {: .2f} SSIM_ALbedo {: .3f} LPIPS_ALBEDO {: .3f}".format(psnr_albedo_linear, ssim_albedo_linear, lpips_albedo_linear))
    print("Evaluating gamma AVG: PSNR_ALBEDO {: .2f} SSIM_ALBEDO {: .3f} LPIPS_ALBEDO {: .3f}".format(psnr_albedo, ssim_albedo, lpips_albedo))
    print("Evaluating AVG: MAE_NORMAL {: .2f}".format(mae_normal))
    # <<< CHANGE START >>>
    print(f"Evaluating AVG: Render Time {avg_render_time_ms:.2f} ms ({fps:.2f} FPS)")
    # <<< CHANGE END >>>
    
    with open(os.path.join(args.model_path, "material_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Results saved to", os.path.join(args.model_path, "material_results.json"))
