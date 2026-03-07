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

from scene.cameras import Camera
import numpy as np
import os
import cv2
import torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        scale = float(resolution_scale * args.resolution)
        resolution = round(orig_w/scale), round(orig_h/scale)
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    HWK = None  # #
    if cam_info.K is not None:
        K = cam_info.K.copy()
        K[:2] = K[:2] / scale
        HWK = (resolution[1], resolution[0], K)

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
        
    if cam_info.mask is not None:
        if cam_info.mask.shape[:2] != resolution[::-1]:
            mask = cv2.resize(cam_info.mask.astype(np.float32), resolution)
        else:
            mask = cam_info.mask.astype(np.float32)
        mask = torch.from_numpy(mask).bool().unsqueeze(0)
    else:
        mask = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, 
                  data_device=args.data_device, HWK=HWK, mask=mask, image_path=cam_info.image_path)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def JSON_to_camera(json_cam):
    rot = np.array(json_cam['rotation'])
    pos = np.array(json_cam['position'])
    W2C = np.zeros((4, 4))
    W2C[:3, :3] = rot
    W2C[:3, 3] = pos
    W2C[3, 3] = 1
    Rt = np.linalg.inv(W2C)
    R = Rt[:3, :3].transpose()
    T = Rt[:3, 3]
    H, W = json_cam['height'], json_cam['width']
    image = torch.zeros(3, H, W)
    if 'cx' not in json_cam:
        if 'fx' in json_cam:
            FovX = focal2fov(json_cam["fx"], W)
            FovY = focal2fov(json_cam["fy"], H)
        else:
            FovX = json_cam["FoVx"]
            FovY = json_cam["FoVy"]
        camera = Camera(colmap_id=0, R=R, T=T, FoVx=FovX, FoVy=FovY, 
                        image=image, image_name=json_cam['img_name'], 
                        uid=json_cam['id'], data_device='cuda',
                        gt_alpha_mask=None)
    else:
        camera = Camera(colmap_id=0, R=R, T=T, FoVx=None, FoVy=None, 
                        image=image, image_name=json_cam['img_name'],
                        uid=json_cam['id'], data_device='cuda',
                        gt_alpha_mask=None)
    return camera