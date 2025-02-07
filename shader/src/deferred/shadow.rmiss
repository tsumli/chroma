#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : require

#include "raytracing_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_payload;

void main() {
    ray_payload.color = vec3(0.0, 0.0, 0.0);
    // ray_payload.dist = -1.0;
    // ray_payload.normal = vec3(0.0, 0.0, 0.0);
    // ray_payload.reflector = 0.0f;
    // ray_payload.shadowed = false;
}
