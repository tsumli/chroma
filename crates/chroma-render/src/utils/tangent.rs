use nalgebra_glm::{
    Vec2,
    Vec3,
};

/// Compute tangent for each vertex and store it in the given vertices.
/// Before calling this function, make sure that the vertices and indices are valid.
/// Vertices should have position, normal, and uv. Indices should have 3 indices per triangle.
/// See "Foundations of Game Engine Development, Volume 2: Rendering", Eric Lengyel, 2019.
/// Section 7.5 for more details.
pub fn compute_tangent(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    // validation
    assert_eq!(positions.len(), normals.len());
    assert_eq!(positions.len(), uvs.len());
    assert_eq!(indices.len() % 3, 0);

    // compute
    let mut tangents = vec![Vec3::zeros(); positions.len()];
    let mut bitangents = vec![Vec3::zeros(); positions.len()];
    for index in indices.chunks(3) {
        let p0 = Vec3::from_row_slice(&positions[index[0] as usize]);
        let p1 = Vec3::from_row_slice(&positions[index[1] as usize]);
        let p2 = Vec3::from_row_slice(&positions[index[2] as usize]);

        let edge1 = p1 - p0;
        let edge2 = p2 - p0;

        let uv0 = Vec2::from_row_slice(&uvs[index[0] as usize]);
        let uv1 = Vec2::from_row_slice(&uvs[index[1] as usize]);
        let uv2 = Vec2::from_row_slice(&uvs[index[2] as usize]);

        let delta_uv1 = uv1 - uv0;
        let delta_uv2 = uv2 - uv0;

        let f = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y);

        let tangent = f * (delta_uv2.y * edge1 - delta_uv1.y * edge2);
        let bitangent = f * (delta_uv2.x * edge1 - delta_uv1.x * edge2);

        tangents[index[0] as usize] += tangent;
        tangents[index[1] as usize] += tangent;
        tangents[index[2] as usize] += tangent;

        bitangents[index[0] as usize] += bitangent;
        bitangents[index[1] as usize] += bitangent;
        bitangents[index[2] as usize] += bitangent;
    }

    let mut tangents_handedness = vec![[0.0f32; 4]; positions.len()];
    for vertex_i in 0..positions.len() {
        let n = Vec3::from_row_slice(&normals[vertex_i]);
        let t = tangents[vertex_i];
        let b = bitangents[vertex_i];

        // Gram-Schmidt orthogonalize
        let t = (t - t.dot(&n) / n.dot(&n) * n).normalize();
        // Calculate handedness
        let handedness = if n.cross(&t).dot(&b) < 0.0 { -1.0 } else { 1.0 };
        let w = handedness;

        tangents_handedness[vertex_i][0] = t.x;
        tangents_handedness[vertex_i][1] = t.y;
        tangents_handedness[vertex_i][2] = t.z;
        tangents_handedness[vertex_i][3] = w;
    }

    tangents_handedness
}
