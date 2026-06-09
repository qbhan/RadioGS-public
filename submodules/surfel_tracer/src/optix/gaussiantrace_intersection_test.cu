#include "auxiliary.h"
#include <optix.h>

#include "gaussiantrace_intersection_test.h"

namespace surfel_tracer {

extern "C" {
	__constant__ Gaussiantrace_intersection_test::Params params;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();

	glm::vec3 ray_o = params.ray_origins[idx.x];
	glm::vec3 ray_d = params.ray_directions[idx.x];

	unsigned int p0;
	optixTrace(
		params.handle,
		make_float3(ray_o.x, ray_o.y, ray_o.z),
		make_float3(ray_d.x, ray_d.y, ray_d.z),
		FLT_EPSILON,                // Min intersection distance
		T_SCENE_MAX,               // Max intersection distance
		0.0f,                // rayTime -- used for motion blur
		OptixVisibilityMask(255), // Specify always visible
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		0,                   // SBT offset
		1,                   // SBT stride
		0,                   // missSBTIndex
		p0
	);

	params.intersection[idx.x] = (bool)p0;
}

extern "C" __global__ void __miss__ms() {
	optixSetPayload_0((uint32_t)0);
}

extern "C" __global__ void __closesthit__ch() {
	optixSetPayload_0((uint32_t)1);
}

}

extern "C" __global__ void __anyhit__ah() {
	// optixTerminateRay();
}