use super::mipmap::generate_mipmap;
use crate::common::{
    buffer::Buffer,
    command_buffer::{
        begin_single_time_command,
        end_single_time_command,
    },
    image_buffer::ImageBuffer,
};
use anyhow::Result;
use ash::vk;
use chroma_scene::image::{
    Image,
    Pixels,
};

#[derive(Clone)]
pub struct Texture {
    image_buffer: ImageBuffer,
}

impl Texture {
    pub fn new(
        images: Vec<Image>,
        image_create_info: vk::ImageCreateInfo,
        view_type: vk::ImageViewType,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        instance: ash::Instance,
    ) -> Result<Self> {
        let image_count = images.len();

        const ASPECT_MASK: vk::ImageAspectFlags = vk::ImageAspectFlags::COLOR;

        let image_buffer = ImageBuffer::new(
            &image_create_info,
            view_type,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ASPECT_MASK,
            physical_device.clone(),
            device.clone(),
            instance.clone(),
        )?;

        let mip_levels = image_create_info.mip_levels;

        // transfer image data to image buffer
        let command_buffer = begin_single_time_command(command_pool, device.clone());
        {
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .src_stage_mask(vk::PipelineStageFlags2KHR::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_TRANSFER)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image_buffer.image())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(mip_levels)
                        .layer_count(image_count as u32)
                        .base_array_layer(0)
                        .base_mip_level(0),
                );
            unsafe {
                device.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfoKHR::default().image_memory_barriers(&[barrier]),
                );
            }
        }

        let mut staging_buffers = Vec::new();
        for (image_i, image) in images.into_iter().enumerate() {
            let staging_buffer = if let Pixels::U8(pixels) = image.pixels {
                Buffer::new(
                    pixels.as_ptr() as *const std::ffi::c_void,
                    std::mem::size_of::<u8>() as u64,
                    pixels.len() as u64,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    physical_device,
                    device.clone(),
                    instance.clone(),
                )
            } else if let Pixels::F32(pixels) = image.pixels {
                Buffer::new(
                    pixels.as_ptr() as *const std::ffi::c_void,
                    std::mem::size_of::<f32>() as u64,
                    pixels.len() as u64,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    physical_device,
                    device.clone(),
                    instance.clone(),
                )
            } else {
                unimplemented!();
            };

            // copy
            let region = vk::BufferImageCopy::default()
                .image_extent(
                    vk::Extent3D::default()
                        .width(image.width)
                        .height(image.height)
                        .depth(1),
                )
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1)
                        .base_array_layer(image_i as u32),
                );

            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    staging_buffer.vk_buffer(),
                    image_buffer.image(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }

            staging_buffers.push(staging_buffer); // to extend lifetime
        }

        // transition
        {
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .src_stage_mask(vk::PipelineStageFlags2KHR::TRANSFER)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::FRAGMENT_SHADER)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image_buffer.image())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(mip_levels)
                        .layer_count(image_count as u32),
                );
            unsafe {
                device.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfoKHR::default().image_memory_barriers(&[barrier]),
                );
            }
        }
        end_single_time_command(command_buffer, queue, command_pool, device.clone());

        // generate mipmaps
        if image_create_info.mip_levels > 1 {
            // transition layout while generating mipmaps
            let image_barrier = vk::ImageMemoryBarrier2::default()
                .src_access_mask(vk::AccessFlags2KHR::empty())
                .dst_access_mask(vk::AccessFlags2KHR::TRANSFER_WRITE)
                .src_stage_mask(vk::PipelineStageFlags2KHR::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::BLIT)
                .old_layout(image_create_info.initial_layout)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image_buffer.image())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(ASPECT_MASK)
                        .base_mip_level(0)
                        .level_count(image_create_info.mip_levels)
                        .base_array_layer(0)
                        .layer_count(image_create_info.array_layers),
                );
            let command_buffer = begin_single_time_command(command_pool, device.clone());
            unsafe {
                device.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfoKHR::default().image_memory_barriers(&[image_barrier]),
                );
            }
            generate_mipmap(
                image_buffer.image(),
                image_create_info.extent.width,
                image_create_info.extent.height,
                image_create_info.mip_levels,
                image_create_info.format,
                command_buffer,
                physical_device,
                device.clone(),
                &instance,
            )?;
            end_single_time_command(command_buffer, queue, command_pool, device.clone());
        }

        Ok(Self { image_buffer })
    }

    pub fn image_view(&self) -> vk::ImageView {
        let image_buffer = &self.image_buffer;
        image_buffer.image_view()
    }
}
