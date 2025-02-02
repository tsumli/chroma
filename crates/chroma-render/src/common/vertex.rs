use crate::common::buffer::{
    copy_buffer,
    Buffer,
};
use ash::vk;
use nalgebra_glm::{
    Vec2,
    Vec3,
    Vec4,
};
use std::mem::offset_of;

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
#[allow(dead_code)]
pub struct Vertex {
    pub position: Vec3,
    pub uv: Vec2,
    pub normal: Vec3,
    pub tangent: Vec4,
    pub color: Vec4,
}

impl Vertex {
    pub fn get_binding_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)]
    }

    pub fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            // position
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            // uv
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, uv) as u32), // offset by size of position
            // normal
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, normal) as u32), // offset by size of position + uv
            // tangent
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(3)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(offset_of!(Vertex, tangent) as u32), /* offset by size of position + uv
                                                              * + normal */
            // color
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(4)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32), /* offset by size of position + uv
                                                            * + normal + tangent */
        ]
    }
}

pub struct VertexBuffer {
    buffer: Buffer,
}

impl VertexBuffer {
    pub fn new(
        vertices: Vec<Vertex>,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        instance: ash::Instance,
    ) -> Self {
        let buffer_size = (vertices.len() * std::mem::size_of::<Vertex>()) as vk::DeviceSize;
        assert!(buffer_size > 0);

        let staging_buffer = Buffer::new(
            vertices.as_ptr() as *const std::ffi::c_void,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            physical_device,
            device.clone(),
            instance.clone(),
        );

        let buffer = Buffer::new(
            std::ptr::null(),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            physical_device,
            device.clone(),
            instance.clone(),
        );

        copy_buffer(
            staging_buffer.vk_buffer(),
            buffer.vk_buffer(),
            buffer_size,
            command_pool,
            queue,
            device.clone(),
        );

        Self { buffer }
    }

    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer.vk_buffer()
    }
}
