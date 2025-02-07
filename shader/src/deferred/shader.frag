#version 460

#extension GL_ARB_shading_language_include : require

#include "frag_input.glsl"
#include "common.glsl"

// uniform
layout(set = 1, binding = 0) uniform material_ubo {
    MaterialParams material;
};

layout(location = 0) in FragInput frag_input;

layout(set = 0, binding = 1) uniform sampler2D color_sampler;
layout(set = 0, binding = 2) uniform sampler2D normal_sampler;
layout(set = 0, binding = 3) uniform sampler2D metallic_roughness_sampler;
layout(set = 0, binding = 4) uniform sampler2D emissive_sampler;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_metallic_roughness;
layout(location = 3) out vec4 out_emissive;

void main() {
    const vec2 texcoord = frag_input.texcoord;

    // Color
    vec4 texture_color = texture(color_sampler, texcoord);
    out_color = texture_color * frag_input.color * material.base_color_factor;

    // Normal
    vec3 normal_ts = texture(normal_sampler, texcoord).xyz;
    normal_ts = normalize(normal_ts * 2.0f - 1.0f);
    mat3x3 tangent_frame_ws =
        mat3x3(normalize(frag_input.tangent_ws), normalize(frag_input.bitangent_ws),
            normalize(frag_input.normal_ws));
    vec3 normal_ws = tangent_frame_ws * normal_ts;
    out_normal = vec4(normal_ws, out_color.a);

    // Metallic Roughness
    vec2 metallic_roughness = texture(metallic_roughness_sampler, texcoord).rg;
    float metallic = metallic_roughness.r * material.metallic_roughness_transmission_factor.r;
    float roughness = metallic_roughness.g * material.metallic_roughness_transmission_factor.g;
    out_metallic_roughness = vec4(metallic, roughness, 0.0, out_color.a);

    // Emissive
    vec3 emissive = texture(emissive_sampler, texcoord).rgb;
    out_emissive = vec4(emissive, 1.0);
}
