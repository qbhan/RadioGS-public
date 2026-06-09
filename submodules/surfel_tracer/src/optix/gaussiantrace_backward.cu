#include "gaussiantrace_backward.h"
#include <optix.h>


namespace surfel_tracer {

extern "C" {
	__constant__ Gaussiantrace_backward::Params params;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	float O_final = params.alpha[idx.x];
	if (O_final==0.0f) return;

	glm::vec3 ray_o = params.ray_origins[idx.x], dL_dray_o = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 ray_d = params.ray_directions[idx.x], dL_dray_d = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 ray_origin;

	glm::vec3 C = glm::vec3(0.0f, 0.0f, 0.0f), C_final = params.color[idx.x], grad_color = params.grad_color[idx.x];
	glm::vec3 N = glm::vec3(0.0f, 0.0f, 0.0f), N_final = params.normal[idx.x], grad_normal = params.grad_normal[idx.x];
	float F[MAX_FEATURE_SIZE] = {0.0f}, F_final[MAX_FEATURE_SIZE], grad_feature[MAX_FEATURE_SIZE];
	const int S = params.S;
	for (int i = 0; i < S; ++i){
		F_final[i] = params.feature[idx.x * S + i];
		grad_feature[i] = params.grad_feature[idx.x * S + i];
	}

	float D = 0.0f, D_final = params.depths[idx.x], grad_depths = params.grad_depths[idx.x];
	float O = 0.0f, grad_alpha = params.grad_alpha[idx.x];

	float T = 1.0f, t_start = 0.0f, t_curr = 0.0f;

	HitInfo hitArray[MAX_BUFFER_SIZE];
	unsigned int hitArrayPtr0 = (unsigned int)((uintptr_t)(&hitArray) & 0xFFFFFFFF);
    unsigned int hitArrayPtr1 = (unsigned int)(((uintptr_t)(&hitArray) >> 32) & 0xFFFFFFFF);

	while ((t_start < T_SCENE_MAX) && (T > params.transmittance_min)){
		ray_origin = ray_o + t_start * ray_d;
		
		for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
			hitArray[i].t = 1e16f;
			hitArray[i].primIdx = -1;
		}
		optixTrace(
			params.handle,
			make_float3(ray_origin.x, ray_origin.y, ray_origin.z),
			make_float3(ray_d.x, ray_d.y, ray_d.z),
			0.0f,                // Min intersection distance
			T_SCENE_MAX,               // Max intersection distance
			0.0f,                // rayTime -- used for motion blur
			OptixVisibilityMask(255), // Specify always visible
			OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
			0,                   // SBT offset
			1,                   // SBT stride
			0,                   // missSBTIndex
			hitArrayPtr0,
			hitArrayPtr1
		);

		for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
			int primIdx = hitArray[i].primIdx;

			if (primIdx == -1) {
				t_curr = T_SCENE_MAX;
				break;
			}
			else{
				t_curr = hitArray[i].t;
				int gs_idx = params.gs_idxs[primIdx];

				float o = params.opacity[gs_idx];
				glm::vec3 mean3D = params.means3D[gs_idx];
				glm::vec3 n = params.normals[gs_idx];
				glm::vec3 ru = params.ru[gs_idx];
				glm::vec3 rv = params.rv[gs_idx];
				
				float cos = -dot(ray_d, n);
				float multiplier = cos > 0 ? 1: -1;
				if ((multiplier < 0) && params.back_culling) continue;

				// glm::vec3 n_flip = multiplier * n;
				glm::vec3 n_flip = n;

				// Compute intersection point
				glm::vec3 ray_o_mean3D = ray_o - mean3D;
				float o_g = glm::dot(n, ray_o_mean3D);
				float d_g = glm::dot(n, ray_d);
				float d = -o_g * d_g / max(1e-6f, d_g * d_g);

				glm::vec3 pos = ray_o + d * ray_d - mean3D;
				glm::vec2 p_g = {glm::dot(ru, pos), glm::dot(rv, pos)}; 

				float G = __expf(-0.5f * glm::dot(p_g, p_g));
				float alpha = min(0.99, o * G);
				if (alpha<params.alpha_min) continue;

				glm::vec3 c = computeColorFromSH_forward(params.deg, ray_d, params.shs + gs_idx * params.max_coeffs);

				float w = T * alpha;
				C += w * c;
				N += w * n_flip;
				D += w * d;
				O += w;
				
				for (int j = 0; j < params.S; ++j){
					F[j] += w * params.features[gs_idx * S + j];
				}

				T *= (1 - alpha);

				glm::vec3 dL_dc = grad_color * w;
				glm::vec3 dL_dnormal = multiplier * grad_normal * w;
				float dL_dfeature[MAX_FEATURE_SIZE];
				for (int j = 0; j < S; ++j){
					dL_dfeature[j] = grad_feature[j] * w;
				}
				float dL_dd = grad_depths * w;
				float dL_dalpha = (
					glm::dot(grad_color, T * c - (C_final - C)) +
					glm::dot(grad_normal, T * n_flip - (N_final - N)) +
					grad_depths * (T * d - (D_final - D)) + 
					grad_alpha * (1 - O_final)
				);

				for (int j = 0; j < S; ++j){
					dL_dalpha += grad_feature[j] * (T * params.features[gs_idx * S + j] - (F_final[j] - F[j]));
				}

				dL_dalpha /= (1 - alpha);
				computeColorFromSH_backward(params.deg, ray_d, params.shs + gs_idx * params.max_coeffs, dL_dc, dL_dray_d, params.grad_shs + gs_idx * params.max_coeffs);
				float dL_do = dL_dalpha * G;
				float dL_dG = dL_dalpha * o;
				glm::vec2 dL_dpg = -dL_dG * G * p_g;
				glm::vec3 dL_dru = dL_dpg.x * pos;
				glm::vec3 dL_drv = dL_dpg.y * pos;
				glm::vec3 dL_dpos = dL_dpg.x * ru + dL_dpg.y * rv;
				
				glm::vec3 dL_dmean3D = -dL_dpos;

				dL_dd += glm::dot(dL_dpos, ray_d);
				dL_dray_o += dL_dpos;
				dL_dray_d += d * dL_dpos;

				float dL_dog = -dL_dd / d_g;
				float dL_ddg = dL_dd * o_g / max(1e-6f, d_g*d_g);
				dL_dray_o += dL_dog * n;
				dL_dray_d += dL_ddg * n;

				dL_dnormal += dL_ddg * ray_d + dL_dog * ray_o_mean3D;
				dL_dmean3D -= dL_dog * n;

        		atomic_add((float*)(params.grad_means3D+gs_idx), dL_dmean3D);
				atomicAdd(params.grad_opacity+gs_idx, dL_do);

        		atomic_add((float*)(params.grad_ru+gs_idx), dL_dru);
        		atomic_add((float*)(params.grad_rv+gs_idx), dL_drv);
        		atomic_add((float*)(params.grad_normals+gs_idx), dL_dnormal);
				for (int j = 0; j < S; ++j){
					atomicAdd(params.grad_features+gs_idx*S+j, dL_dfeature[j]);
				}

				if (T < params.transmittance_min){
					break;
				}
			}
		}
		t_start += t_curr;
	}
	atomic_add((float*)(params.grad_rays_o+idx.x), dL_dray_o);
	atomic_add((float*)(params.grad_rays_d+idx.x), dL_dray_d);
}

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __closesthit__ch() {
}

extern "C" __global__ void __anyhit__ah() {
    HitInfo* hitArray = (HitInfo*)((uintptr_t)optixGetPayload_0() | ((uintptr_t)optixGetPayload_1() << 32));

	float THit = optixGetRayTmax();
    int i_prim = optixGetPrimitiveIndex();
	HitInfo newHit = {THit, i_prim};

	for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
		if (hitArray[i].primIdx == -1){
			hitArray[i] = newHit;
			break;
		}
        else if (hitArray[i].t > newHit.t) {
			host_device_swap<HitInfo>(hitArray[i], newHit);
        }
    }

	if (THit < hitArray[MAX_BUFFER_SIZE - 1].t) {
        optixIgnoreIntersection(); 
    }

}

}
