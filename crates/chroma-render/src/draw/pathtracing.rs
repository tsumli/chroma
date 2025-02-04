use crate::{
    common::{
        self,
        camera::TransformParams,
        descriptor_pool,
        image_buffer::ImageBuffer,
        pipeline_layout,
        uniform_buffer,
    },
    texture::texture::Texture,
};
use ash::vk::{
    self,
};
use std::collections::HashMap;

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RenderTarget {
    Output,
}

#[allow(dead_code)]
pub struct Pathtracing {
    transform_ubo: uniform_buffer::UniformBuffer<TransformParams>,
    material_ubo: Vec<uniform_buffer::UniformBuffer<common::material::MaterialParams>>,
    vertex_buffers: Vec<common::vertex::VertexBuffer>,
    index_buffers: Vec<common::index::IndexBuffer>,
    base_color_textures: Vec<Texture>,
    texture_samplers: Vec<vk::Sampler>,
    graphics_framebuffers: Vec<common::framebuffer::Framebuffer>,
    command_pool: common::command_pool::CommandPool,
    render_pass: common::render_pass::RenderPass,
    ash_device: ash::Device,
    rasterize_pipelines: Vec<common::graphics_pipeline::GraphicsPipeline>,
    rasterize_pipeline_layouts: Vec<pipeline_layout::PipelineLayout>,
    rasterize_descriptor_pool: descriptor_pool::DescriptorPool,
    rasterize_descriptor_sets: Vec<common::descriptor_set::DescriptorSet>,
    render_targets: HashMap<RenderTarget, ImageBuffer>,
}
