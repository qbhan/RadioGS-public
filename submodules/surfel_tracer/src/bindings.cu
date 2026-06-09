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

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <surfel_tracer/bvh.h>

namespace py = pybind11;
namespace surfel_tracer {

class GaussianTracer {
public:
    GaussianTracer(){
        triangle_bvh = TriangleBvhBase::make();
    }

    void build_bvh(const torch::Tensor& triangles){
        const size_t n_triangles = triangles.size(0);
        cudaStream_t m_stream = at::cuda::getCurrentCUDAStream();;
        triangle_bvh->build_bvh(triangles.data_ptr<float>(), n_triangles, m_stream);
    }

    void update_bvh(const torch::Tensor& triangles){
        const size_t n_triangles = triangles.size(0);
        cudaStream_t m_stream = at::cuda::getCurrentCUDAStream();;
        triangle_bvh->update_bvh(triangles.data_ptr<float>(), n_triangles, m_stream);
    }

    void trace_forward(
        const torch::Tensor rays_o, const torch::Tensor rays_d, const torch::Tensor gs_idxs, 
        const torch::Tensor means3D, const torch::Tensor opacity, const torch::Tensor ru, const torch::Tensor rv, const torch::Tensor normals, const torch::Tensor features, const torch::Tensor shs, 
        torch::Tensor color, torch::Tensor normal, torch::Tensor feature, torch::Tensor depth, torch::Tensor alpha, 
        const float alpha_min, const float transmittance_min, const int deg, const bool back_culling
        ){
        const uint32_t n_elements = rays_o.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        int max_coeffs = shs.size(1);
        int S = features.size(1);

        triangle_bvh->gaussian_trace_forward(
            n_elements, S, (const glm::vec3*)rays_o.data_ptr<float>(), (const glm::vec3*)rays_d.data_ptr<float>(), gs_idxs.data_ptr<int>(), 
            (const glm::vec3*)means3D.data_ptr<float>(), opacity.data_ptr<float>(), (const glm::vec3*)ru.data_ptr<float>(), (const glm::vec3*)rv.data_ptr<float>(), (const glm::vec3*)normals.data_ptr<float>(), features.data_ptr<float>(), (const glm::vec3*)shs.data_ptr<float>(), 
            (glm::vec3*)color.data_ptr<float>(), (glm::vec3*)normal.data_ptr<float>(), feature.data_ptr<float>(), depth.data_ptr<float>(), alpha.data_ptr<float>(), 
            alpha_min, transmittance_min, deg, max_coeffs, back_culling, stream);
    }
    
    void intersection_test(
        const torch::Tensor rays_o, const torch::Tensor rays_d, const torch::Tensor gs_idxs, 
        const torch::Tensor means3D, const torch::Tensor opacity, const torch::Tensor ru, const torch::Tensor rv, const torch::Tensor normals, torch::Tensor intersection
        ){
        const uint32_t n_elements = rays_o.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->intersection_test(
            n_elements, (const glm::vec3*)rays_o.data_ptr<float>(), (const glm::vec3*)rays_d.data_ptr<float>(), gs_idxs.data_ptr<int>(), 
            (const glm::vec3*)means3D.data_ptr<float>(), opacity.data_ptr<float>(), (const glm::vec3*)ru.data_ptr<float>(), (const glm::vec3*)rv.data_ptr<float>(), (const glm::vec3*)normals.data_ptr<float>(), intersection.data_ptr<bool>(), stream);
    }
    
    void trace_backward(
        const torch::Tensor rays_o, const torch::Tensor rays_d, const torch::Tensor gs_idxs, 
        const torch::Tensor means3D, const torch::Tensor opacity, const torch::Tensor ru, const torch::Tensor rv, const torch::Tensor normals, const torch::Tensor features, const torch::Tensor shs, 
        const torch::Tensor color, const torch::Tensor normal, const torch::Tensor feature, const torch::Tensor depth, const torch::Tensor alpha, 
        torch::Tensor grad_rays_o, torch::Tensor grad_rays_d, torch::Tensor grad_means3D, torch::Tensor grad_opacity, torch::Tensor grad_ru, torch::Tensor grad_rv, torch::Tensor grad_normals, torch::Tensor grad_features, torch::Tensor grad_shs, 
        const torch::Tensor grad_out_color, const torch::Tensor grad_out_normal, const torch::Tensor grad_out_feature, const torch::Tensor grad_out_depth, const torch::Tensor grad_out_alpha,
        const float alpha_min, const float transmittance_min, const int deg, const bool back_culling
        ){
        const uint32_t n_elements = rays_o.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        int max_coeffs = shs.size(1);
        int S = features.size(1);

        triangle_bvh->gaussian_trace_backward(
            n_elements, S, (const glm::vec3*)rays_o.data_ptr<float>(), (const glm::vec3*)rays_d.data_ptr<float>(), gs_idxs.data_ptr<int>(), 
            (const glm::vec3*)means3D.data_ptr<float>(), opacity.data_ptr<float>(), (const glm::vec3*)ru.data_ptr<float>(), (const glm::vec3*)rv.data_ptr<float>(), (const glm::vec3*)normals.data_ptr<float>(), features.data_ptr<float>(), (const glm::vec3*)shs.data_ptr<float>(), 
            (const glm::vec3*)color.data_ptr<float>(), (const glm::vec3*)normal.data_ptr<float>(), feature.data_ptr<float>(), depth.data_ptr<float>(), alpha.data_ptr<float>(), 
            (glm::vec3*)grad_rays_o.data_ptr<float>(), (glm::vec3*)grad_rays_d.data_ptr<float>(), (glm::vec3*)grad_means3D.data_ptr<float>(), grad_opacity.data_ptr<float>(), (glm::vec3*)grad_ru.data_ptr<float>(), (glm::vec3*)grad_rv.data_ptr<float>(), (glm::vec3*)grad_normals.data_ptr<float>(), grad_features.data_ptr<float>(), (glm::vec3*)grad_shs.data_ptr<float>(), 
            (const glm::vec3*)grad_out_color.data_ptr<float>(), (const glm::vec3*)grad_out_normal.data_ptr<float>(), grad_out_feature.data_ptr<float>(), grad_out_depth.data_ptr<float>(), grad_out_alpha.data_ptr<float>(),
            alpha_min, transmittance_min, deg, max_coeffs, back_culling, stream);
    }

    std::shared_ptr<TriangleBvhBase> triangle_bvh;
};

GaussianTracer* create_gaussiantracer() {
    return new GaussianTracer{};
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

py::class_<surfel_tracer::GaussianTracer>(m, "GaussianTracer")
    .def("intersection_test", &surfel_tracer::GaussianTracer::intersection_test)
    .def("trace_forward", &surfel_tracer::GaussianTracer::trace_forward)
    .def("trace_backward", &surfel_tracer::GaussianTracer::trace_backward)
    .def("build_bvh", &surfel_tracer::GaussianTracer::build_bvh)
    .def("update_bvh", &surfel_tracer::GaussianTracer::update_bvh);

m.def("create_gaussiantracer", &surfel_tracer::create_gaussiantracer);

}