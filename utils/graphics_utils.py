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
import math
import numpy as np
from typing import NamedTuple
import torch.nn.functional as F
import cv2


def fibonacci_sphere_sampling(normals, sample_num, random_rotate=True):
    pre_shape = normals.shape[:-1]
    if len(pre_shape) > 1:
        normals = normals.reshape(-1, 3)
    delta = np.pi * (3.0 - np.sqrt(5.0))

    # fibonacci sphere sample around z axis
    idx = torch.arange(sample_num, dtype=torch.float, device='cuda')[None]
    z = (1 - 2 * idx / (2 * sample_num - 1)).clamp_min(np.sin(10/180*np.pi))
    rad = torch.sqrt(1 - z ** 2)
    theta = delta * idx
    if random_rotate:
        theta = torch.rand(*pre_shape, 1, device='cuda') * 2 * np.pi + theta
    y = torch.cos(theta) * rad
    x = torch.sin(theta) * rad
    z_samples = torch.stack([x, y, z.expand_as(y)], dim=-2)

    # rotate to normal
    # z_vector = torch.zeros_like(normals)
    # z_vector[..., 2] = 1  # [H, W, 3]
    # rotation_matrix = rotation_between_vectors(z_vector, normals)
    rotation_matrix = rotation_between_z(normals)
    incident_dirs = rotation_matrix @ z_samples
    incident_dirs = F.normalize(incident_dirs, dim=-2).transpose(-1, -2)
    incident_areas = torch.ones_like(incident_dirs)[..., 0:1] * 2 * np.pi
    if len(pre_shape) > 1:
        incident_dirs = incident_dirs.reshape(*pre_shape, sample_num, 3)
        incident_areas = incident_areas.reshape(*pre_shape, sample_num, 1)
    return incident_dirs, incident_areas

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

# #
def getProjectionMatrixCorrect(znear, zfar, H, W, K):

  top = (K[1,2])/K[1,1] * znear
  bottom = -(H - K[1,2])/K[1,1] * znear
  right = (K[0,2])/K[0,0] * znear
  left = -(W - K[0,2])/K[0,0] * znear

  P = torch.zeros(4, 4)

  z_sign = 1.0

  P[0, 0] = 2.0 * znear / (right - left)
  P[1, 1] = 2.0 * znear / (top - bottom)
  P[0, 2] = (right + left) / (right - left)
  P[1, 2] = (top + bottom) / (top - bottom)
  P[3, 2] = z_sign
  P[2, 2] = z_sign * zfar / (zfar - znear)
  P[2, 3] = -(zfar * znear) / (zfar - znear)
  return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def rotation_between_z(vec):
    """
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    Args:
        vec: [..., 3]

    Returns:
        R: [..., 3, 3]

    """
    v1 = -vec[..., 1]
    v2 = vec[..., 0]
    v3 = torch.zeros_like(v1)
    v11 = v1 * v1
    v22 = v2 * v2
    v33 = v3 * v3
    v12 = v1 * v2
    v13 = v1 * v3
    v23 = v2 * v3
    cos_p_1 = (vec[..., 2] + 1).clamp_min(1e-7)
    R = torch.zeros(vec.shape[:-1] + (3, 3,), dtype=torch.float32, device="cuda")
    R[..., 0, 0] = 1 + (-v33 - v22) / cos_p_1
    R[..., 0, 1] = -v3 + v12 / cos_p_1
    R[..., 0, 2] = v2 + v13 / cos_p_1
    R[..., 1, 0] = v3 + v12 / cos_p_1
    R[..., 1, 1] = 1 + (-v33 - v11) / cos_p_1
    R[..., 1, 2] = -v1 + v23 / cos_p_1
    R[..., 2, 0] = -v2 + v13 / cos_p_1
    R[..., 2, 1] = v1 + v23 / cos_p_1
    R[..., 2, 2] = 1 + (-v22 - v11) / cos_p_1
    R = torch.where((vec[..., 2] + 1 > 0)[..., None, None], R,
                    -torch.eye(3, dtype=torch.float32, device="cuda").expand_as(R))
    return R

def rgb_to_srgb(img, clip=True):
    # hdr img
    if isinstance(img, np.ndarray):
        # assert len(img.shape) == 3, img.shape
        # assert img.shape[2] == 3, img.shape
        img = np.where(img > 0.0031308, np.power(np.maximum(img, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * img)
        if clip:
            img = np.clip(img, 0.0, 1.0)
        return img
    elif isinstance(img, torch.Tensor):
        # assert len(img.shape) == 3, img.shape
        # assert img.shape[0] == 3, img.shape
        img = torch.where(img > 0.0031308, torch.pow(torch.max(img, torch.tensor(0.0031308)), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * img)
        if clip:
            img = torch.clamp(img, 0.0, 1.0)
        return img
    else:
        raise TypeError("Unsupported input type. Supported types are numpy.ndarray and torch.Tensor.")


def srgb_to_rgb(img):
    # f is LDR
    if isinstance(img, np.ndarray):
        img = np.where(img <= 0.04045, img / 12.92, np.power((np.maximum(img, 0.04045) + 0.055) / 1.055, 2.4))
        return img
    elif isinstance(img, torch.Tensor):
        img = torch.where(img <= 0.04045, img / 12.92, torch.pow((torch.max(img, torch.tensor(0.04045)) + 0.055) / 1.055, 2.4))
        return img
    else:
        raise TypeError("Unsupported input type. Supported types are numpy.ndarray and torch.Tensor.")
    
def orthonormal_basis(n):
    # build t, b so (t,b,n) is an ONB for each row of n: (B,3)
    helper = torch.where(
        n.abs()[...,2:3] < 0.999,
        torch.tensor([0,0,1.], device=n.device),
        torch.tensor([1,0,0.], device=n.device),
    )
    t = F.normalize(torch.cross(helper.expand_as(n), n), dim=-1)
    b = torch.cross(n, t)
    return t, b

def to_world(local, n):
    t, b = orthonormal_basis(n)
    return local[...,0:1]*t + local[...,1:2]*b + local[...,2:3]*n

def sample_ggx(normals, view_dirs, roughness, sample_num):
    """
    normals:    (B,3)
    view_dirs:  (B,3)
    roughness:  (B,)      # in [0,1]
    sample_num: int S

    Returns:
      l_spec:  (B,S,3)
      pdf:     (B,S)
      F_spec:  (B,S)
    """
    B, device = normals.shape[0], normals.device
    # 1) draw uniforms in [0,1]
    u1 = torch.rand(B, sample_num, device=device)
    u2 = torch.rand(B, sample_num, device=device)

    # 2) compute alpha and its square
    alpha = roughness**2              # (B,)
    alpha2 = alpha**2                 # (B,)

    # 3) sample half‐angle θh, φh
    phi = 2 * math.pi * u2            # (B,S)
    denom = 1 + (alpha2.unsqueeze(1) - 1) * u1
    cos_h = torch.sqrt((1 - u1) / denom).clamp(0,1)   # (B,S)
    sin_h = torch.sqrt((1 - cos_h*cos_h).clamp(0,1))  # (B,S)

    # 4) local half‐vectors h in tangent space
    local_h = torch.stack([
        sin_h * torch.cos(phi),
        sin_h * torch.sin(phi),
        cos_h
    ], dim=-1)                    # (B,S,3)

    # 5) world half‐vector H via ONB
    t, b = orthonormal_basis(normals)            # (B,3),(B,3)
    t = t.unsqueeze(1);  b = b.unsqueeze(1)
    n = normals.unsqueeze(1)
    H = local_h[...,0:1]*t + local_h[...,1:2]*b + local_h[...,2:3]*n  # (B,S,3)
    H = F.normalize(H, dim=-1)

    # 6) reflect view → L
    V = view_dirs.unsqueeze(1)                   # (B,S,3)
    VdotH = (V*H).sum(-1,keepdim=True).clamp(min=1e-6)  # (B,S,1)
    L = 2*VdotH*H - V                             # (B,S,3)

    # 7) GGX D(h) and pdf
    NdotH = (n*H).sum(-1,keepdim=True).clamp(min=1e-6)    # (B,S,1)
    denomD = (NdotH*NdotH*(alpha2.unsqueeze(1).unsqueeze(2)-1) + 1)
    D = (alpha2.unsqueeze(1).unsqueeze(2)) / (math.pi * denomD*denomD)
    pdf = (D * NdotH / (4 * VdotH))                        # (B,S, 1)

    # 8) Schlick Fresnel
    F0 = 0.04
    F_spec = (F0 + (1 - F0) * (1 - VdotH)**5).squeeze(-1)  # (B,S)

    return L, pdf, F_spec

def ggx_pdf(diff_dirs, normals, view_dirs, roughness):
    """
    Compute the GGX specular sampling PDF for a batch of given directions.
    
    Args:
      diff_dirs: (B, S, 3)  — candidate incoming directions
      normals:   (B, 3)
      view_dirs: (B, 3)
      roughness: (B,)       — scalar roughness per sample
    
    Returns:
      pdf:       (B, S)     — p_s(l) for each direction
    """
    B, S, _ = diff_dirs.shape
    device = diff_dirs.device

    # 1) lift N and V to (B,1,3) so they broadcast to (B,S,3)
    N = normals.unsqueeze(1)      # (B,1,3)
    V = view_dirs.unsqueeze(1)    # (B,1,3)

    # 2) half‐vector H
    H = F.normalize(V + diff_dirs, dim=-1)  # (B,S,3)

    # 3) dot products, clamped
    n_dot_h = torch.clamp((N * H).sum(-1, keepdim=True), min=1e-6)  # (B,S,1)
    v_dot_h = torch.clamp((V * H).sum(-1, keepdim=True), min=1e-6)  # (B,S,1)

    # 4) alpha² shaped (B,1,1)
    # alpha2 = (roughness ** 2).unsqueeze(1).unsqueeze(2)            # (B,1,1)
    alpha2 = (roughness ** 2)
    alpha2 = (alpha2 ** 2).unsqueeze(1).unsqueeze(2) # roughness**4

    # 5) GGX D(h)
    denom = (n_dot_h * n_dot_h * (alpha2 - 1) + 1)                 # (B,S,1)
    D = alpha2 / (math.pi * denom * denom)                         # (B,S,1)

    # 6) final specular PDF
    pdf = (D * n_dot_h) / (4 * v_dot_h)                            # (B,S,1)

    return pdf  # → (B,S)

def random_hemisphere_sampling(normals, sample_num):
    pre_shape = normals.shape[:-1]
    if len(pre_shape) > 1:
        normals = normals.reshape(-1, 3)

    # Randomly sample directions on hemisphere around z-axis
    u = torch.rand(*pre_shape, sample_num, device='cuda')
    v = torch.rand(*pre_shape, sample_num, device='cuda')

    theta = 2 * np.pi * u
    phi = torch.acos(v)

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    z_samples = torch.stack([x, y, z], dim=-2)

    # Rotate to align with provided normals
    rotation_matrix = rotation_between_z(normals)
    incident_dirs = rotation_matrix @ z_samples
    incident_dirs = F.normalize(incident_dirs, dim=-2).transpose(-1, -2)

    if len(pre_shape) > 1:
        incident_dirs = incident_dirs.reshape(*pre_shape, sample_num, 3)

    return incident_dirs

def env_map_to_cam_to_world_by_convention(envmap: np.ndarray, c2w, convention):
    R = c2w[:3,:3]
    H, W = envmap.shape[:2]
    theta, phi = np.meshgrid(np.linspace(-0.5*np.pi, 1.5*np.pi, W), np.linspace(0., np.pi, H))
    # theta, phi = np.meshgrid(np.linspace(1.0 * np.pi, -1.0 * np.pi, W), np.linspace(0., np.pi, H))
    viewdirs = np.stack([-np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)],
                        axis=-1).reshape(H*W, 3)    # [H, W, 3]
    # theta, phi = np.meshgrid(np.linspace(1.0 * np.pi, -1.0 * np.pi, W), np.linspace(0., np.pi, H))
    # viewdirs = np.stack([np.cos(theta) * np.sin(phi),
    #                     np.sin(theta) * np.sin(phi),
    #                     np.cos(phi)], axis=-1).reshape(H*W, 3)    # [H, W, 3]
    viewdirs = (R.T @ viewdirs.T).T.reshape(H, W, 3)
    viewdirs = viewdirs.reshape(H, W, 3)
    # This is correspond to the convention of +Z at left, +Y at top
    # -np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)
    coord_y = ((np.arccos(viewdirs[..., 1])/np.pi*(H-1)+H)%H).astype(np.float32)
    coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
    envmap_remapped = cv2.remap(envmap, coord_x, coord_y, cv2.INTER_LINEAR)

    if convention == 'ours':
        return envmap_remapped
    if convention == 'physg':
        # change convention from ours (Left +Z, Up +Y) to physg (Left -Z, Up +Y)
        envmap_remapped_physg = np.roll(envmap_remapped, W//2, axis=1)
        return envmap_remapped_physg
    if convention == 'nerd':
        # change convention from ours (Left +Z-X, Up +Y) to nerd (Left +Z+X, Up +Y)
        envmap_remapped_nerd = envmap_remapped[:,::-1,:]
        return envmap_remapped_nerd

    assert convention == 'invrender', convention
    # change convention from ours (Left +Z-X, Up +Y) to invrender (Left -X+Y, Up +Z)
    theta, phi = np.meshgrid(np.linspace(1.0 * np.pi, -1.0 * np.pi, W), np.linspace(0., np.pi, H))
    viewdirs = np.stack([np.cos(theta) * np.sin(phi),   
                        np.sin(theta) * np.sin(phi),
                        np.cos(phi)], axis=-1)    # [H, W, 3]
    # viewdirs = np.stack([-viewdirs[...,0], viewdirs[...,2], viewdirs[...,1]], axis=-1)
    coord_y = ((np.arccos(viewdirs[..., 1])/np.pi*(H-1)+H)%H).astype(np.float32)
    coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
    envmap_remapped_Inv = cv2.remap(envmap_remapped, coord_x, coord_y, cv2.INTER_LINEAR)
    return envmap_remapped_Inv

def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)


# used for spherical initialization for DiscreteSDF
def normal2quat(v2):
    v1 = torch.zeros_like(v2)
    v1[:, 2] = 1
    a  = torch.cross(v1, v2)
    w = torch.sqrt((v1**2).sum(-1) * (v2**2).sum(-1)) + (v1 * v2).sum(-1)
    q = torch.stack([w, a[:, 0],  a[:, 1], a[:, 2]], -1)
    q = F.normalize(q, dim=-1)
    return q