use anyhow::Result;
use ash::vk;

pub fn generate_mipmap(
    image: vk::Image,
    width: u32,
    height: u32,
    mip_level: u32,
    format: vk::Format,
    command_buffer: vk::CommandBuffer,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    instance: &ash::Instance,
) -> Result<()> {
    let props =
        unsafe { instance.get_physical_device_format_properties(physical_device.clone(), format) };
    anyhow::ensure!(
        props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
            && props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::BLIT_SRC)
            && props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::BLIT_DST),
        "texture image format is invalid!"
    );

    let mut barrier = vk::ImageMemoryBarrier2::default()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1),
        );

    for mip_i in 1..mip_level {
        barrier.subresource_range.base_mip_level = mip_i - 1;
        barrier.src_access_mask = vk::AccessFlags2KHR::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags2KHR::TRANSFER_READ;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_stage_mask = vk::PipelineStageFlags2KHR::BLIT;
        barrier.dst_stage_mask = vk::PipelineStageFlags2KHR::ALL_TRANSFER;

        unsafe {
            device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfoKHR::default().image_memory_barriers(&[barrier]),
            );
        }

        let image_blits = [vk::ImageBlit2::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(mip_i - 1)
                    .layer_count(1),
            )
            .src_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: (width >> (mip_i - 1)) as i32,
                    y: (height >> (mip_i - 1)) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(mip_i)
                    .layer_count(1),
            )
            .dst_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: (width >> mip_i) as i32,
                    y: (height >> mip_i) as i32,
                    z: 1,
                },
            ])];

        let blit_image_info = vk::BlitImageInfo2::default()
            .src_image(image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(&image_blits);

        unsafe {
            device.cmd_blit_image2(command_buffer, &blit_image_info);
        }

        barrier.src_access_mask = vk::AccessFlags2KHR::TRANSFER_READ;
        barrier.dst_access_mask = vk::AccessFlags2KHR::SHADER_READ;
        barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_stage_mask = vk::PipelineStageFlags2KHR::ALL_TRANSFER;
        barrier.dst_stage_mask = vk::PipelineStageFlags2KHR::FRAGMENT_SHADER;

        unsafe {
            device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfoKHR::default().image_memory_barriers(&[barrier]),
            );
        }
    }

    barrier.subresource_range.base_mip_level = mip_level - 1;
    barrier.src_access_mask = vk::AccessFlags2KHR::TRANSFER_WRITE;
    barrier.dst_access_mask = vk::AccessFlags2KHR::SHADER_READ;
    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    barrier.src_stage_mask = vk::PipelineStageFlags2KHR::ALL_TRANSFER;
    barrier.dst_stage_mask = vk::PipelineStageFlags2KHR::FRAGMENT_SHADER;

    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfoKHR::default().image_memory_barriers(&[barrier]),
        );
    }

    Ok(())
}
