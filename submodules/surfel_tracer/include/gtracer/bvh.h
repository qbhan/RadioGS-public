#pragma once
#include <iostream>
#include <string>
#include <vector>

#include <cstdint>
#include <cmath>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <glm/glm.hpp>
#include <gtracer/gpu_memory.h>

#include <memory>

namespace gtracer {

class TriangleBvhBase {
public:
    TriangleBvhBase() {};
    virtual void intersection_test(
        uint32_t n_elements, const glm::vec3* rays_o, const glm::vec3* rays_d, const int* gs_idxs, 
        const glm::vec3* means3D, const float* opacity, const glm::vec3* ru, const glm::vec3* rv, const glm::vec3* normals, bool* intersection, cudaStream_t stream) = 0;
    virtual void gaussian_trace_forward(
        uint32_t n_elements, const int S, const glm::vec3* rays_o, const glm::vec3* rays_d, const int* gs_idxs, 
        const glm::vec3* means3D, const float* opacity, const glm::vec3* ru, const glm::vec3* rv, const glm::vec3* normals, const float* features, const glm::vec3* shs, 
        glm::vec3* color, glm::vec3* normal, float* feature, float* depth, float* alpha, 
        const float alpha_min, const float transmittance_min, const int deg, const int max_coeffs, const bool back_culling, cudaStream_t stream) = 0;
    virtual void gaussian_trace_backward(
        uint32_t n_elements, const int S, const glm::vec3* rays_o, const glm::vec3* rays_d, const int* gs_idxs, 
        const glm::vec3* means3D, const float* opacity, const glm::vec3* ru,const glm::vec3* rv, const glm::vec3* normals, const float* features, const glm::vec3* shs, 
        const glm::vec3* color, const glm::vec3* normal, const float* feature, const float* depth, const float* alpha, 
        glm::vec3* grad_rays_o, glm::vec3* grad_rays_d, glm::vec3* grad_means3D, float* grad_opacity, glm::vec3* grad_ru, glm::vec3* grad_rv, glm::vec3* grad_normals, float* grad_features, glm::vec3* grad_shs, 
        const glm::vec3* grad_color, const glm::vec3* grad_normal, const float* grad_feature, const float* grad_depth, const float* grad_alpha,
        const float alpha_min, const float transmittance_min, const int deg, const int max_coeffs, const bool back_culling, cudaStream_t stream) = 0;

    virtual void build_bvh(const float* triangles, int n_triangles, cudaStream_t stream) = 0;
    virtual void update_bvh(const float* triangles, int n_triangles, cudaStream_t stream) = 0;

    static std::unique_ptr<TriangleBvhBase> make();
};

}