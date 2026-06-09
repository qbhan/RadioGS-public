import torch
from surfel_tracer import _C


class _GaussianTrace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bvh, rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, alpha_min, transmittance_min, deg, back_culling):    
        color = torch.zeros_like(rays_o)
        normal = torch.zeros_like(rays_o)
        feature = torch.zeros(*rays_o.shape[:-1], features.shape[-1], device=rays_o.device, dtype=rays_o.dtype)
        depth = torch.zeros_like(rays_o[:, 0])
        alpha = torch.zeros_like(rays_o[:, 0])
        bvh.trace_forward(
            rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, 
            color, normal, feature, depth, alpha, 
            alpha_min, transmittance_min, deg, back_culling
        )
        
        ctx.alpha_min = alpha_min
        ctx.transmittance_min = transmittance_min
        ctx.deg = deg
        ctx.bvh = bvh
        ctx.back_culling = back_culling
        ctx.save_for_backward(rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, color, normal, feature, depth, alpha)
        return color, normal, feature, depth, alpha

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_normal, grad_out_feature, grad_out_depth, grad_out_alpha):
        rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, color, normal, feature, depth, alpha = ctx.saved_tensors
        grad_rays_o = torch.zeros_like(rays_o)
        grad_rays_d = torch.zeros_like(rays_d)
        grad_means3D = torch.zeros_like(means3D)
        grad_opacity = torch.zeros_like(opacity)
        grad_ru = torch.zeros_like(ru)
        grad_rv = torch.zeros_like(rv)
        grad_normals = torch.zeros_like(normals)
        grad_features = torch.zeros_like(features)
        grad_shs = torch.zeros_like(shs)
        
        ctx.bvh.trace_backward(
            rays_o, rays_d, gs_idxs, means3D, opacity, ru, rv, normals, features, shs, 
            color, normal, feature, depth, alpha, 
            grad_rays_o, grad_rays_d, grad_means3D, grad_opacity, grad_ru, grad_rv, grad_normals, grad_features, grad_shs,
            grad_out_color, grad_out_normal, grad_out_feature, grad_out_depth, grad_out_alpha,
            ctx.alpha_min, ctx.transmittance_min, ctx.deg, ctx.back_culling
        )
        
        grads = (
            None,
            grad_rays_o,
            grad_rays_d,
            None,
            grad_means3D,
            grad_opacity,
            grad_ru,
            grad_rv,
            grad_normals,
            grad_features,
            grad_shs,
            None,
            None,
            None,
            None,
        )

        return grads


class GaussianTracer():
    def __init__(self, transmittance_min=0.001):
        self.impl = _C.create_gaussiantracer()
        self.transmittance_min = transmittance_min
        
    def build_bvh(self, vertices_b, faces_b, gs_idxs):
        self.faces_b = faces_b
        self.gs_idxs = gs_idxs.int()
        self.impl.build_bvh(vertices_b[faces_b])

    def update_bvh(self, vertices_b, faces_b, gs_idxs):
        assert (self.faces_b == faces_b).all(), "Update bvh must keep the triangle id not change~"
        self.gs_idxs = gs_idxs.int()
        self.impl.update_bvh(vertices_b[faces_b])

    def trace(self, rays_o, rays_d, means3D, opacity, ru, rv, normals, features, shs, alpha_min, deg=3, back_culling=False):
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        means3D = means3D.contiguous()
        opacity = opacity.contiguous()
        ru = ru.contiguous()
        rv = rv.contiguous()
        normals = normals.contiguous()
        if features is not None:
            features = features.contiguous()
        else:
            features = torch.zeros_like(means3D[:, :0])
        shs = shs.contiguous()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        B = rays_o.shape[0]
        mask = torch.zeros(B, dtype=torch.bool, device='cuda')
        self.impl.intersection_test(rays_o, rays_d, self.gs_idxs, means3D, opacity, ru, rv, normals, mask)
        color = torch.zeros(B, 3, dtype=torch.float32, device='cuda')
        normal = torch.zeros(B, 3, dtype=torch.float32, device='cuda')
        feature = torch.zeros(B, features.shape[-1], dtype=torch.float32, device='cuda')
        depth = torch.zeros(B, dtype=torch.float32, device='cuda')
        alpha = torch.zeros(B, dtype=torch.float32, device='cuda')
        
        rays_o_ = rays_o[mask]
        rays_d_ = rays_d[mask]
        if not rays_o_.shape[0] == 0:
            color[mask], normal[mask], feature[mask], depth[mask], alpha[mask] = _GaussianTrace.apply(self.impl, rays_o_, rays_d_, self.gs_idxs, means3D, opacity, ru, rv, normals, features, shs, alpha_min, self.transmittance_min, deg, back_culling)

        color = color.view(*prefix, 3)
        normal = normal.view(*prefix, 3)
        feature = feature.view(*prefix, features.shape[-1])
        depth = depth.view(*prefix)
        alpha = alpha.view(*prefix)
        
        return color, normal, feature, depth, alpha