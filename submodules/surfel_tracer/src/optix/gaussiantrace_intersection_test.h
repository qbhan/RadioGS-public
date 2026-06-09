#pragma once
#include "auxiliary.h"

#include <optix.h>

namespace surfel_tracer {

struct Gaussiantrace_intersection_test {
	struct Params {
		const glm::vec3* ray_origins;
		const glm::vec3* ray_directions;
		const int* gs_idxs;
		const glm::vec3* means3D;
		const float* opacity;
		const glm::vec3* ru;
		const glm::vec3* rv;
		const glm::vec3* normals;
		bool* intersection;
		OptixTraversableHandle handle;
	};

	struct RayGenData {};
	struct MissData {};
	struct HitGroupData {};
};

}
