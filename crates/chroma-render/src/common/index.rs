use crate::common::buffer::copy_buffer;
use ash::vk;

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct Index(pub u32);

impl Index {
    pub fn get_type() -> vk::IndexType {
        vk::IndexType::UINT32
    }
}

// Implement `From<Index>` for usize for ergonomic conversion
impl From<Index> for usize {
    fn from(index: Index) -> Self {
        index.0 as usize
    }
}

#[allow(dead_code)]
pub struct IndexBuffer {
    buffer: super::buffer::Buffer,
    len: usize,
}

#[allow(dead_code)]
impl IndexBuffer {
    pub fn new(
        indices: Vec<Index>,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        instance: ash::Instance,
    ) -> Self {
        let buffer_size = (indices.len() * std::mem::size_of::<Index>()) as vk::DeviceSize;
        assert!(buffer_size > 0);

        let staging_buffer = super::buffer::Buffer::new(
            indices.as_ptr() as *const std::ffi::c_void,
            buffer_size,
            1,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            physical_device,
            device.clone(),
            instance.clone(),
        );

        let buffer = super::buffer::Buffer::new(
            std::ptr::null(),
            buffer_size,
            1,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
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

        let len = indices.len();

        Self { buffer, len }
    }

    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer.vk_buffer()
    }

    pub fn len(&self) -> usize {
        self.len
    }
}
