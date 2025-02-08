#[derive(Clone, Copy, Debug, Default)]
pub struct RaytracingPipelineProperties {
    pub shader_group_handle_size: u32,
    pub shader_group_base_alignment: u32,
    pub shader_group_handle_alignment: u32,
}
