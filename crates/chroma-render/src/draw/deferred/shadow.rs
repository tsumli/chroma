use super::raytracing::ShaderBindingTable;
use crate::common::{
    descriptor_pool::DescriptorPool,
    descriptor_set::DescriptorSet,
    image_buffer::ImageBuffer,
    pipeline_layout::PipelineLayout,
    raytracing_pipeline::RaytracingPipeline,
};
use ash::vk;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShadowRenderTarget {
    Output,
}

pub const NUM_SHADOW_DESCRIPTOR_SETS: usize = 3;

#[derive(Clone)]
pub struct ShadowResource {
    pub _bottom_level_acceleration_structures: Vec<crate::common::buffer::Buffer>,
    pub _bottom_level_acceleration_structure_handles: Vec<vk::AccelerationStructureKHR>,
    pub _top_level_acceleration_structure: crate::common::buffer::Buffer,
    pub _top_level_acceleration_structure_handle: vk::AccelerationStructureKHR,
    pub _shadow_descriptor_pool: DescriptorPool,
    pub shadow_descriptor_sets: Vec<DescriptorSet>,
    pub shadow_pipelines: Vec<RaytracingPipeline>,
    pub shadow_pipeline_layouts: Vec<PipelineLayout>,
    pub shadow_render_targets: HashMap<ShadowRenderTarget, ImageBuffer>,
    pub shadow_shader_binding_tables: Vec<ShaderBindingTable>,
}
