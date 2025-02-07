#ifndef COMMON
#define COMMON

struct TransformParams {
    mat4x4 world;
    mat4x4 view;
    mat4x4 proj;
    mat4x4 view_proj;
    mat4x4 world_view_proj;
    mat4x4 proj_to_world;
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

#endif
