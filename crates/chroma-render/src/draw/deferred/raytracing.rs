use crate::common::buffer::Buffer;
use ash::vk;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GeometryNode {
    pub vertex_buffer_device_address: vk::DeviceAddress,
    pub index_buffer_device_address: vk::DeviceAddress,
}

#[derive(Clone)]
pub struct ShaderBindingTable {
    pub raygen: Buffer,
    pub hit: Buffer,
    pub miss: Buffer,
}

pub fn create_identity_transform_matrix() -> vk::TransformMatrixKHR {
    vk::TransformMatrixKHR {
        matrix: [
            1.0, 0.0, 0.0, 0.0, // First column
            0.0, 1.0, 0.0, 0.0, // Second column
            0.0, 0.0, 1.0, 0.0, // Third column
        ],
    }
}
