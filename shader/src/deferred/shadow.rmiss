#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : require

#include "raytracing_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_payload;

void main() {
    ray_payload.shadowed = false;
}
