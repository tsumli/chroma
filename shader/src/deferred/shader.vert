#version 460

#extension GL_ARB_shading_language_include : require

#include "frag_input.glsl"

// uniform
struct TransformParams {
    mat4x4 world;
    mat4x4 view;
    mat4x4 proj;
    mat4x4 view_proj;
    mat4x4 world_view_proj;
    mat4x4 proj_to_world;
};
layout(set = 0, binding = 0) uniform ubo {
    TransformParams transform;
};

// vertex
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec4 tangent;
layout(location = 4) in vec4 color;

// out
layout(location = 0) out FragInput frag_input;

void main() {
    // Convert normal
    frag_input.normal_ws = normalize(mat3x3(transform.world) * normal);

    // Reconstruct the tangent frame
    frag_input.tangent_ws = normalize(mat3x3(transform.world) * tangent.xyz);
    frag_input.bitangent_ws =
        normalize(cross(frag_input.normal_ws, frag_input.tangent_ws)) * tangent.w;

    // Calculate the clip-space position
    frag_input.position_cs = transform.world_view_proj * vec4(position, 1.0);

    // Pass through the rest of the data
    frag_input.texcoord = uv;
    frag_input.color = color;

    // Output the clip-space position
    gl_Position = frag_input.position_cs;
}
