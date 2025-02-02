use anyhow::Result;
use ash::vk;
use std::u64;

pub fn create_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    count: u32,
) -> Result<Vec<vk::CommandBuffer>> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(count)
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    let command_buffers =
        unsafe { device.allocate_command_buffers(&command_buffer_allocate_info)? };
    Ok(command_buffers)
}

pub fn begin_single_time_command(
    command_pool: vk::CommandPool,
    device: ash::Device,
) -> vk::CommandBuffer {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    let command_buffer = unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }
        .expect("failed to allocate command buffer")[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .expect("failed to begin command buffer");
    }

    command_buffer
}

pub fn end_single_time_command(
    command_buffer: vk::CommandBuffer,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    device: ash::Device,
) {
    unsafe {
        device
            .end_command_buffer(command_buffer)
            .expect("failed to end command buffer");

        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .expect("failed to submit queue");

        device
            .queue_wait_idle(queue)
            .expect("failed to wait for queue idle");

        device.free_command_buffers(command_pool, &[command_buffer]);
    }
}
