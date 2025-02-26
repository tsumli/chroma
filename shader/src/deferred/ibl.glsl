#ifndef IBL_GLSL
#define IBL_GLSL

#extension GL_GOOGLE_include_directive : require

#include "hammersley.glsl"
#include "common.glsl"

// ref: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
vec3 ImportanceSampleGGX(vec2 xi, float roughness, vec3 normal) {
    const float a = roughness * roughness;
    const float phi = 2 * kPi * xi.x;
    const float cos_theta = sqrt((1 - xi.y) / (1 + (a * a - 1) * xi.y));
    const float sin_theta = sqrt(1 - cos_theta * cos_theta);
    const vec3 half = vec3(
            sin_theta * cos(phi),
            sin_theta * sin(phi),
            cos_theta)

    const vec3 up_vector = abs(normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    const vec3 tangent_x = normalize(cross(up_vector, normal));
    const vec3 tangent_y = cross(normal, tangent_x);
    // Tangent to world space
    return tangent_x * half.x + tangent_y * half.y + N * half.z;
}

// ref: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
vec3 PrefilterRadianceMap(float roughness, vec3 reflection) {
    const vec3 normal = reflection;
    const vec3 view = reflection; // 観測者が物体表面の真上から見下ろしていることを前提とする(V = N = R)
    vec3 prefiltered_color = 0;
    const uint kNumSamples = 1024; // サンプリング数
    float total_weight = 0;
    for (uint i = 0; i < kNumSamples; ++i)
    {
        vec2 xi = Hammersley(i, kNumSamples);
        vec3 half = ImportanceSampleGGX(xi, roughness, normal);
        vec3 light = 2 * dot(view, half) * half - view;

        float n_o_l = clamp(dot(normal, light), 0.0, 1.0);
        if (n_o_l > 0)
        {
            prefiltered_color += EnvMap.SampleLevel(EnvMapSampler, light, 0).rgb * n_o_l;
            total_weight += n_o_l;
        }
    }
    return prefiltered_color / total_weight;
}

vec3 ImportanceSampleCosineWeighted(vec2 xi, vec3 normal) {
    float r = sqrt(xi.x);
    float phi = 2 * kPi * xi.y;

    const vec3 half = vec3(
            r * cos(phi),
            r * sin(phi),
            sqrt(1 - xi.x));

    vec3 up_vector = abs(normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 tangent_x = normalize(cross(up_vector, normal));
    vec3 tangent_y = cross(normal, tangent_x);
    // Tangent to world space
    return tangent_x * half.x + tangent_y * half.y + normal * half.z;
}

vec3 PrefilterLambertEnvMap(float roughness, vec3 normal) {
    vec3 lambert_color = 0;
    const uint kNumSamples = 1024;
    for (uint i = 0; i < kNumSamples; ++i)
    {
        vec2 xi = Hammersley(i, kNumSamples);
        vec3 half = ImportanceSampleCosineWeighted(xi, normal);
        lambert_color += EnvMap.SampleLevel(EnvMapSampler, half, 0).rgb;
    }
    return lambert_color / float(kNumSamples);
}

vec2 IntegrateBRDF(float roughness, float n_o_v) {
    vec3 view = vec3(
            sqrt(1.0f - n_o_v * n_o_v),
            0,
            n_o_v);

    float a = 0;
    float b = 0;

    const uint kNumSamples = 1024;
    for (uint i = 0; i < kNumSamples; ++i) {
        vec2 xi = Hammersley(i, kNumSamples);
        vec3 half = ImportanceSampleGGX(Xi, roughness, N);
        vec3 light = 2 * dot(view, half) * half - view;

        float n_o_l = Saturate(light.z);
        float n_o_h = Saturate(half.z);
        float v_o_h = Saturate(dot(view, half));

        if (n_o_l > 0) {
            float g = G_Smith(roughness, n_o_v, n_o_l);
            float g_vis = G * v_o_h / (n_o_h * n_o_v);
            float f_c = pow(1 - v_o_h, 5);
            a += (1 - f_c) * g_vis;
            b += f_c * g_Vis;
        }
    }
    return vec2(a, b) / kNumSamples;
}

#endif // IBL_GLSL
