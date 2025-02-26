#ifndef BRDF_GLSL
#define BRDF_GLSL

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

float FresnelSchlick(float f0, float f90, float cosine) {
    return f0 + (f90 - f0) * pow(1.0 - cosine, 5.0);
}

vec3 FresnelSchlick(vec3 f0, float f90, float cosine) {
    return f0 + (f90 - f0) * pow(1.0 - cosine, 5.0);
}

vec3 LamberDiffuse(vec3 albedo) {
    return albedo * kInvPi;
}

/// Normalized Disney diffuse BRDF
vec3 BrdfDiffuse(vec3 albedo, float n_o_v, float n_o_l, float l_o_h, float roughness) {
    float bias = mix(0.0, 0.5, roughness);
    float factor = mix(1.0, 1.0 / 1.51, roughness);
    float fd90 = bias + 2.0 * l_o_h * l_o_h * roughness;
    float fl = FresnelSchlick(1.0, fd90, n_o_l);
    float fv = FresnelSchlick(1.0, fd90, n_o_v);
    return albedo * kInvPi * fl * fv * factor;
}

float NormalDistributionGgx(vec3 normal, vec3 half_dir, float roughness) {
    float roughness2 = roughness * roughness;
    float n_o_h = Saturate(dot(normal, half_dir));
    float a = (1.0 - (1.0 - roughness2) * n_o_h * n_o_h);
    return roughness2 * kInvPi / (a * a);
}

float MaskingShadowingSmithJoint(vec3 normal, vec3 view_dir, vec3 light_dir, float roughness) {
    float roughness2 = roughness * roughness;
    float n_o_v = Saturate(dot(normal, view_dir));
    float n_o_l = Saturate(dot(normal, light_dir));
    float lv = 0.5 * (-1.0 + sqrt(1.0 + roughness2 * (1.0 / (n_o_v * n_o_v) - 1.0)));
    float ll = 0.5 * (-1.0 + sqrt(1.0 + roughness2 * (1.0 / (n_o_l * n_o_l) - 1.0)));
    return 1.0 / (1.0 + lv + ll);
}

/// Cook-Torrance specular BRDF
vec3 BrdfSpecular(vec3 albedo, vec3 normal, vec3 view_dir, vec3 light_dir, vec3 half_dir, float roughness) {
    float n_o_v = Saturate(dot(normal, view_dir));
    float n_o_l = Saturate(dot(normal, light_dir));
    float v_o_h = Saturate(dot(view_dir, half_dir));
    float d = NormalDistributionGgx(normal, half_dir, roughness);
    float g = MaskingShadowingSmithJoint(normal, view_dir, light_dir, roughness);
    vec3 f = FresnelSchlick(albedo, 1.0, v_o_h);
    return d * g * f / (4.0 * n_o_v * n_o_l);
}

#endif // BRDF_GLSL
