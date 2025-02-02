const float PI = 3.14159265359f;
const float EPSILON = 1e-6f;
const float INV_PI = 1.0f / PI;

float FresnelSchlick(float f0, float f90, float cosine) {
    return f0 + (f90 - f0) * pow(1.0 - cosine, 5.0);
}

vec3 FresnelSchlick(vec3 f0, float f90, float cosine) {
    return f0 + (f90 - f0) * pow(1.0 - cosine, 5.0);
}

vec3 LamberDiffuse(vec3 albedo) {
    return albedo * INV_PI;
}

/// Normalized Disney diffuse BRDF
vec3 BrdfDiffuse(vec3 albedo, float n_o_v, float n_o_l, float l_o_h, float roughness) {
    float bias = mix(0.0, 0.5, roughness);
    float factor = mix(1.0, 1.0 / 1.51, roughness);
    float fd90 = bias + 2.0 * l_o_h * l_o_h * roughness;
    float fl = FresnelSchlick(1.0, fd90, n_o_l);
    float fv = FresnelSchlick(1.0, fd90, n_o_v);
    return albedo * INV_PI * fl * fv * factor;
}

float NormalDistributionGgx(vec3 normal, vec3 half_dir, float roughness) {
    float roughness2 = roughness * roughness;
    float n_o_h = clamp(dot(normal, half_dir), 0.0, 1.0);
    float a = (1.0 - (1.0 - roughness2) * n_o_h * n_o_h);
    return roughness2 * INV_PI / (a * a);
}

float MaskingShadowingSmithJoint(vec3 normal, vec3 view_dir, vec3 light_dir, float roughness) {
    float roughness2 = roughness * roughness;
    float n_o_v = clamp(dot(normal, view_dir), 0.0, 1.0);
    float n_o_l = clamp(dot(normal, light_dir), 0.0, 1.0);
    float lv = 0.5 * (-1.0 + sqrt(1.0 + roughness2 * (1.0 / (n_o_v * n_o_v) - 1.0)));
    float ll = 0.5 * (-1.0 + sqrt(1.0 + roughness2 * (1.0 / (n_o_l * n_o_l) - 1.0)));
    return 1.0 / (1.0 + lv + ll);
}

/// Cook-Torrance specular BRDF
vec3 BrdfSpecular(vec3 albedo, vec3 normal, vec3 view_dir, vec3 light_dir, vec3 half_dir, float roughness) {
    float n_o_v = clamp(dot(normal, view_dir), 0.0, 1.0);
    float n_o_l = clamp(dot(normal, light_dir), 0.0, 1.0);
    float v_o_h = clamp(dot(view_dir, half_dir), 0.0, 1.0);
    float d = NormalDistributionGgx(normal, half_dir, roughness);
    float g = MaskingShadowingSmithJoint(normal, view_dir, light_dir, roughness);
    vec3 f = FresnelSchlick(albedo, 1.0, v_o_h);
    return d * g * f / (4.0 * n_o_v * n_o_l);
}
