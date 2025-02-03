pub struct MeshletObject {
    pub meshlets: Vec<meshopt::ffi::meshopt_Meshlet>,
    pub meshlet_vertices: Vec<u32>,
    pub meshlet_triangles: Vec<u8>,
}

/// See https://github.com/zeux/meshoptimizer for the original source code.
pub fn generate_meshlets(vertices: Vec<[f32; 3]>, indices: Vec<u32>) -> MeshletObject {
    const MAX_VERTICES: usize = 64;
    const MAX_TRIANGLES: usize = 124;
    const CONE_WEIGHT: f32 = 0.0;
    let num_vertices = vertices.len();
    let num_index = indices.len();
    let max_meshlet =
        unsafe { meshopt::ffi::meshopt_buildMeshletsBound(num_index, MAX_VERTICES, MAX_TRIANGLES) };

    let mut meshlets = vec![
        meshopt::ffi::meshopt_Meshlet {
            vertex_offset: 0,
            vertex_count: 0,
            triangle_offset: 0,
            triangle_count: 0,
        };
        max_meshlet
    ];
    let mut meshlet_vertices = vec![0u32; max_meshlet * MAX_VERTICES];
    let mut meshlet_triangles = vec![0u8; max_meshlet * MAX_TRIANGLES * 3];

    let vertices_flatten = vertices.iter().flatten().copied().collect::<Vec<f32>>();
    let meshlet_count = unsafe {
        meshopt::ffi::meshopt_buildMeshlets(
            meshlets.as_mut_ptr(),
            meshlet_vertices.as_mut_ptr(),
            meshlet_triangles.as_mut_ptr(),
            indices.as_ptr(),
            num_index,
            vertices_flatten.as_ptr(),
            num_vertices,
            std::mem::size_of::<f32>() * 3,
            MAX_VERTICES,
            MAX_TRIANGLES,
            CONE_WEIGHT,
        )
    };

    // truncate
    meshlets.truncate(meshlet_count as usize);
    let last = meshlets.last().unwrap();
    meshlet_vertices.truncate((last.vertex_offset + last.vertex_count) as usize);
    meshlet_triangles.truncate((last.triangle_offset + last.triangle_count) as usize * 3);

    assert!(!meshlets.is_empty());
    assert!(!meshlet_vertices.is_empty());
    assert!(!meshlet_triangles.is_empty());

    MeshletObject {
        meshlets,
        meshlet_vertices,
        meshlet_triangles,
    }
}
