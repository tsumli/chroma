#ifndef FRAG_INPUT
#define FRAG_INPUT

struct FragInput {
    vec4 position_cs;
    vec4 position_ws;
    vec3 normal_ws;
    vec2 texcoord;
    vec3 tangent_ws;
    vec3 bitangent_ws;
    vec4 color;
};

#endif
