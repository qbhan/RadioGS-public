#pragma once

#include "auxiliary.h"
#include <optix.h>

namespace surfel_tracer {

struct Gaussiantrace_backward {
	struct Params {
		const glm::vec3* ray_origins;
		const glm::vec3* ray_directions;
		const int* gs_idxs;
		const glm::vec3* means3D;
		const float* opacity;
		const glm::vec3* ru;
		const glm::vec3* rv;
		const glm::vec3* normals;
		const float* features;
		const glm::vec3* shs;
		const glm::vec3* color;
		const glm::vec3* normal;
		const float* feature;
		const float* depths;
		const float* alpha;
		glm::vec3* grad_rays_o;
		glm::vec3* grad_rays_d;
		glm::vec3* grad_means3D;
		float* grad_opacity;
		glm::vec3* grad_ru;
		glm::vec3* grad_rv;
		glm::vec3* grad_normals;
		float* grad_features;
		glm::vec3* grad_shs;
		const glm::vec3* grad_color;
		const glm::vec3* grad_normal;
		const float* grad_feature;
		const float* grad_depths;
		const float* grad_alpha;
		float alpha_min;
		float transmittance_min;
		int deg;
		int max_coeffs;
		int S;
		bool back_culling;
		OptixTraversableHandle handle;
	};

	struct RayGenData {};
	struct MissData {};
	struct HitGroupData {};
};

}
