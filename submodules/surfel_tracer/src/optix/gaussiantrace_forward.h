#pragma once
#include "auxiliary.h"

#include <optix.h>

namespace surfel_tracer {

struct Gaussiantrace_forward {
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
		glm::vec3* color;
		glm::vec3* normal;
		float* feature;
		float* depth;
		float* alpha;
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
