import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2

import math

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def sample_gaussian_mask(opacity: torch.Tensor, roughness: torch.Tensor, num_samples: int = 4096, strategy: str='random', replacement: bool = False) -> torch.Tensor:
    """
    Creates a boolean mask that selects `num_samples` gaussians,
    with selection probability proportional to their opacity.
    
    Args:
        opacity (torch.Tensor): Tensor of opacities. It can be any shape where
                                the selection is along a flat view.
        num_samples (int): Number of gaussians to sample. Default is 2^12 = 4096.
        replacement (bool): Whether to sample with replacement (if there are fewer than num_samples gaussians).
    
    Returns:
        mask (torch.Tensor): Boolean tensor with the same shape as `opacity` where True indicates a selected gaussian.
    """
    if strategy == 'opacity':
        # Flatten the tensor to 1D for sampling
        flat_opacity = opacity.contiguous().view(-1) # (num_gaussians,)
        
        # Normalize to create probability distribution
        p = flat_opacity / flat_opacity.sum()
        
        # Sample indices according to the opacity-based probabilities
        indices = torch.multinomial(p, num_samples, replacement=replacement)
        
        # Create a boolean mask and mark the selected indices
        mask = torch.zeros_like(flat_opacity, dtype=torch.bool) # (num_gaussians,)
    elif strategy == 'roughness':
        # Flatten the tensor to 1D for sampling
        flat_roughness = roughness.contiguous().view(-1)
        # select top num-samples roughness indices
        indices = torch.topk(flat_roughness, num_samples, largest=True).indices
        mask = torch.zeros_like(flat_roughness, dtype=torch.bool)

    elif strategy == 'random':
        # Flatten the tensor to 1D for sampling
        flat_opacity = opacity.contiguous().view(-1)
        # Sample random indices
        # indices = torch.randint(0, flat_opacity.shape[0], (num_samples,), device=flat_opacity.device)
        indices = torch.randperm(flat_opacity.shape[0], device=flat_opacity.device)[:num_samples]
        mask = torch.zeros_like(flat_opacity, dtype=torch.bool)

    mask[indices] = True
    
    # # Reshape mask to the original opacity shape
    # mask = mask.view_as(opacity) # (num_pts, num_sample)
    return mask