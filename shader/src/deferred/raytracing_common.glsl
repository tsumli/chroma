#ifndef RAYTRACING_COMMON
#define RAYTRACING_COMMON

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

struct GeometryNode {
    uint64_t vertex_buffer_device_address;
    uint64_t index_buffer_device_address;
};

struct Triangle {
    Vertex vertices[3];
    vec3 position;
    vec3 normal;
    vec2 uv;
    vec4 tangent;
};

struct RayPayload {
    vec3 color;
    bool shadowed;
};

const int kMaxRecursion = 2;

#endif
