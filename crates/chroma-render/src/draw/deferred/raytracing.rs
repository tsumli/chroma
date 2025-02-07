use crate::common::buffer::Buffer;
use ash::vk;

#[derive(Debug, Clone, Copy, Default)]
pub struct GeometryNode {
    pub vertex_buffer_device_address: vk::DeviceAddress,
    pub index_buffer_device_address: vk::DeviceAddress,
}

#[derive(Clone)]
pub struct BindingTables {
    pub raygen: Buffer,
    pub hit: Buffer,
    pub miss: Buffer,
}
