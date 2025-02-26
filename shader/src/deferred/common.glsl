#ifndef COMMON_GLSL
#define COMMON_GLSL

struct TransformParams {
    mat4x4 world;
    mat4x4 view;
    mat4x4 proj;
    mat4x4 view_proj;
    mat4x4 world_view_proj;
    mat4x4 proj_to_world;
    mat4x4 view_inv;
    mat4x4 proj_inv;
};

struct Vertex {
    vec3 position;
    vec2 uv;
    vec3 normal;
    vec4 tangent;
    vec4 color;
};

struct LightParams {
    vec4 pos;
    float range;
    vec4 color;
};

struct MaterialParams {
    vec4 base_color_factor;
    vec4 metallic_roughness_transmission_factor;
};

struct CameraParams {
    vec3 position;
};

const float kEpsilon = 0.001f;
const float kPi = 3.14159265359f;
const float kInvPi = 1.0f / kPi;

// Saturate
float Saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

double Saturate(double x) {
    return clamp(x, 0.0, 1.0);
}

vec2 Saturate(vec2 x) {
    return clamp(x, 0.0, 1.0);
}

vec3 Saturate(vec3 x) {
    return clamp(x, 0.0, 1.0);
}

#endif
