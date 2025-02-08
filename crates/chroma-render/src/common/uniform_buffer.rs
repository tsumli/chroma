use super::{
    buffer::Buffer,
    consts::MAX_FRAMES_IN_FLIGHT,
};
use ash::vk;

#[derive(Clone)]
pub struct UniformBuffer<T> {
    buffers: Vec<Buffer>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> UniformBuffer<T> {
    pub fn new(
        data: T,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        instance: ash::Instance,
    ) -> Self {
        let mut buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        let type_size = std::mem::size_of::<T>() as vk::DeviceSize;
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer = Buffer::new(
                &data as *const _ as *const std::ffi::c_void,
                type_size,
                1,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                physical_device,
                device.clone(),
                instance.clone(),
            );
            buffers.push(buffer);
        }
        Self {
            buffers,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn vk_buffer(&self, frame_index: usize) -> vk::Buffer {
        self.buffers[frame_index].vk_buffer()
    }

    pub fn update(&self, frame_index: usize, data: T) {
        self.buffers[frame_index].update_buffer(&data as *const _ as *const std::ffi::c_void);
    }

    pub fn type_size(&self) -> vk::DeviceSize {
        std::mem::size_of::<T>() as vk::DeviceSize
    }
}
