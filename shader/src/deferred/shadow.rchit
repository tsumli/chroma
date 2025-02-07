#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require

#include "common.glsl"
#include "raytracing_common.glsl"

layout(location = 2) rayPayloadEXT bool shadowed;

hitAttributeEXT vec2 attribs;

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1) uniform ubo_transform {
    TransformParams transform;
};
layout(set = 2, binding = 0) buffer GeometryNodes {
    GeometryNode nodes[];
} geometry_nodes;

layout(buffer_reference, scalar) buffer Vertices {
    vec4 v[];
};
layout(buffer_reference, scalar) buffer Indices {
    uint i[];
};
layout(buffer_reference, scalar) buffer Data {
    vec4 f[];
};
layout(location = 0) rayPayloadInEXT RayPayload ray_payload;

// This function will unpack our vertex buffer data into a single triangle and calculates uv
// coordinates
Triangle UnpackTriangle(const uint primitive_index) {
    Triangle tri;
    GeometryNode geometry_node = geometry_nodes.nodes[gl_InstanceID];
    Indices indices = Indices(geometry_node.index_buffer_device_address);
    Vertices vertices = Vertices(geometry_node.vertex_buffer_device_address);

    // Unpack vertices
    // Data is packed as vec4 so we can map to the glTF vertex structure from the host side
    const uint tri_index = primitive_index * 3;
    for (uint i = 0; i < 3; i++) {
        const uint vertex_offset =
            uint(indices.i[tri_index + i]) * 3; // 12 bytes = 3 vec4 per Vertex
        const vec4 d0 = vertices.v[vertex_offset + 0]; // pos.xyz, normal.x
        const vec4 d1 = vertices.v[vertex_offset + 1]; // normal.yz, uv.xy
        const vec4 d2 = vertices.v[vertex_offset + 2]; // tangent.xyzw
        tri.vertices[i].position = d0.xyz;
        tri.vertices[i].normal = vec3(d0.w, d1.xy);
        tri.vertices[i].uv = d1.zw;
        tri.vertices[i].tangent = d2;
    }
    // Calculate values at barycentric coordinates
    vec3 barycentric_coords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    tri.position = tri.vertices[0].position * barycentric_coords.x +
            tri.vertices[1].position * barycentric_coords.y +
            tri.vertices[2].position * barycentric_coords.z;
    tri.normal = tri.vertices[0].normal * barycentric_coords.x +
            tri.vertices[1].normal * barycentric_coords.y +
            tri.vertices[2].normal * barycentric_coords.z;
    tri.uv = tri.vertices[0].uv * barycentric_coords.x + tri.vertices[1].uv * barycentric_coords.y +
            tri.vertices[2].uv * barycentric_coords.z;
    tri.tangent = tri.vertices[0].tangent * barycentric_coords.x +
            tri.vertices[1].tangent * barycentric_coords.y +
            tri.vertices[2].tangent * barycentric_coords.z;
    return tri;
}

void main() {
    Triangle tri = UnpackTriangle(gl_PrimitiveID);
    GeometryNode geometry_node = geometry_nodes.nodes[gl_InstanceID];
    const vec3 normal_ws = normalize(mat3x3(transform.world) * tri.normal);

    ray_payload.dist = gl_RayTmaxEXT;
    ray_payload.normal = normal_ws;
    ray_payload.reflector = 1.0f;

    vec3 light_pos = vec3(5.0, 5.0, 5.0);

    // Shadow casting
    float t_min = 0.01;
    float t_max = length(light_pos - tri.position.xyz);
    float epsilon = 0.001;
    vec3 origin =
        gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT + tri.normal * epsilon;

    // Trace shadow ray and offset indices to match shadow hit/miss shader group indices
    shadowed = true;
    traceRayEXT(tlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT |
            gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF, 0, 0, 1, origin, t_min, normalize(light_pos - tri.position.xyz), t_max, 2);

    if (shadowed) {
        ray_payload.color *= 0.3;
    }
}
