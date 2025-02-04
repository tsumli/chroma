use ash::vk;

pub struct ScratchBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    device_address: vk::DeviceAddress,
}
