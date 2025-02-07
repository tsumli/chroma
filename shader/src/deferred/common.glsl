#ifndef COMMON
#define COMMON

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

#endif
