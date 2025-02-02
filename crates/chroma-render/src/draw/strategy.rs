use crate::common::image_buffer::ImageBuffer;
use anyhow::Result;
use ash::vk;

pub trait DrawStrategy {
    fn draw(&self, command_buffer: vk::CommandBuffer, image_index: u32) -> Result<()>;
    fn output_render_target(&self) -> &ImageBuffer;
}
