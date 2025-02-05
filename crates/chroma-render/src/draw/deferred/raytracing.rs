use ash::vk;

pub struct GeometryNode {
    pub vertex_buffer_device_address: vk::DeviceAddress,
    pub index_buffer_device_address: vk::DeviceAddress,
}
