use super::strategy::DrawStrategy;
use crate::common::{
    image_buffer::ImageBuffer,
    swapchain::Swapchain,
};
use anyhow::Result;
use ash::vk;

pub struct Simple {
    pub output_render_target: ImageBuffer,
}

impl Simple {
    pub fn new(
        swapchain: &Swapchain,
        physical_device: vk::PhysicalDevice,
        ash_device: ash::Device,
        instance: ash::Instance,
        _entry: ash::Entry,
    ) -> Result<Self> {
        let output_render_target = ImageBuffer::new(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(swapchain.vk_swapchain_format())
                .extent(
                    vk::Extent3D::default()
                        .width(swapchain.vk_swapchain_extent().width)
                        .height(swapchain.vk_swapchain_extent().height)
                        .depth(1),
                )
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .array_layers(1)
                .mip_levels(1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .samples(vk::SampleCountFlags::TYPE_1),
            vk::ImageViewType::TYPE_2D,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::ImageAspectFlags::COLOR,
            physical_device,
            ash_device.clone(),
            instance.clone(),
        )?;
        Ok(Self {
            output_render_target,
        })
    }
}

impl DrawStrategy for Simple {
    fn draw(&self, _command_buffer: vk::CommandBuffer, _image_index: u32) -> Result<()> {
        Ok(())
    }

    fn output_render_target(&self) -> &ImageBuffer {
        &self.output_render_target
    }
}
