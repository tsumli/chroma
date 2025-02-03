use super::memory::find_memory_type;
use ash::vk;

fn flush_mapped_memory(
    device: &ash::Device,
    device_memory: vk::DeviceMemory,
    size: vk::DeviceSize,
) {
    let mapped_range = [vk::MappedMemoryRange::default()
        .memory(device_memory)
        .size(size)];
    unsafe {
        device.flush_mapped_memory_ranges(&mapped_range).unwrap();
    }
}

pub fn copy_buffer(
    src: vk::Buffer,
    dst: vk::Buffer,
    size: vk::DeviceSize,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    device: ash::Device,
) {
    let command_buffer =
        super::command_buffer::begin_single_time_command(command_pool, device.clone());

    let copy_region = vk::BufferCopy::default()
        .size(size)
        .src_offset(0)
        .dst_offset(0);

    unsafe {
        device.cmd_copy_buffer(command_buffer, src, dst, &[copy_region]);
    }

    super::command_buffer::end_single_time_command(command_buffer, queue, command_pool, device);
}

#[derive(Clone)]
pub struct Buffer {
    buffer: vk::Buffer,
    device: ash::Device,
    device_memory: vk::DeviceMemory,
    memory_property_flags: vk::MemoryPropertyFlags,
    mapped_memory: *mut std::ffi::c_void,
    type_size: vk::DeviceSize,
    len: vk::DeviceSize,
}

impl Buffer {
    pub fn new(
        data: *const std::ffi::c_void,
        type_size: vk::DeviceSize,
        len: vk::DeviceSize,
        usage_flags: vk::BufferUsageFlags,
        memory_property_flags: vk::MemoryPropertyFlags,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        instance: ash::Instance,
    ) -> Self {
        let size = type_size * len;
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&create_info, None).unwrap() };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let mut alloc_flags_info =
            if usage_flags.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                vk::MemoryAllocateFlagsInfo::default()
                    .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS)
                    .device_mask(0)
            } else {
                vk::MemoryAllocateFlagsInfo::default()
            };

        let memory_type = find_memory_type(
            mem_requirements.memory_type_bits,
            memory_property_flags,
            physical_device,
            instance,
        )
        .unwrap();

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut alloc_flags_info);

        let device_memory = unsafe { device.allocate_memory(&alloc_info, None).unwrap() };

        let mut mapped_memory = std::ptr::null_mut();
        if data != std::ptr::null() {
            mapped_memory = unsafe {
                device
                    .map_memory(device_memory, 0, size, vk::MemoryMapFlags::empty())
                    .unwrap()
            };
            if !memory_property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
                flush_mapped_memory(&device, device_memory, size);
            }

            unsafe {
                std::ptr::copy(data, mapped_memory, size as usize);
                device.unmap_memory(device_memory);
            }
        }

        unsafe {
            device.bind_buffer_memory(buffer, device_memory, 0).unwrap();
        }

        Self {
            buffer,
            device,
            device_memory,
            memory_property_flags,
            mapped_memory,
            type_size,
            len,
        }
    }

    pub fn update_buffer(&self, data: *const std::ffi::c_void) {
        assert!(self.len == 1);
        if self.mapped_memory.is_null() || data.is_null() {
            log::warn!("Buffer is not mappable or data is null");
            return;
        }

        unsafe {
            std::ptr::copy(data, self.mapped_memory, self.type_size as usize);
        }

        if !self
            .memory_property_flags
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            flush_mapped_memory(&self.device, self.device_memory, self.type_size);
        }
    }

    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn len(&self) -> vk::DeviceSize {
        self.len
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.type_size * self.len
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.device_memory, None);
        }
    }
}

pub fn allocate_device_local_buffer(
    data: *const std::ffi::c_void,
    type_size: vk::DeviceSize,
    len: vk::DeviceSize,
    usage_flags: vk::BufferUsageFlags,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    instance: ash::Instance,
) -> Buffer {
    let size = type_size * len;
    assert!(size > 0);

    let staging_buffer = Buffer::new(
        data,
        type_size,
        len,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        physical_device,
        device.clone(),
        instance.clone(),
    );

    let buffer = Buffer::new(
        std::ptr::null(),
        type_size,
        len,
        vk::BufferUsageFlags::TRANSFER_DST | usage_flags,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        physical_device,
        device.clone(),
        instance.clone(),
    );

    copy_buffer(
        staging_buffer.vk_buffer(),
        buffer.vk_buffer(),
        size,
        command_pool,
        queue,
        device.clone(),
    );

    buffer
}
