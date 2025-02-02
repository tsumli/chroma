use anyhow::Result;
use ash::vk;

#[derive(Clone)]
pub struct RenderPass {
    vk_render_pass: ash::vk::RenderPass,
    device: ash::Device,
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_render_pass(self.vk_render_pass, None);
        }
    }
}

impl RenderPass {
    pub fn new(
        render_pass_create_info: vk::RenderPassCreateInfo,
        device: ash::Device,
    ) -> Result<Self> {
        let vk_render_pass = unsafe { device.create_render_pass(&render_pass_create_info, None)? };
        Ok(Self {
            vk_render_pass,
            device,
        })
    }

    pub fn vk_render_pass(&self) -> vk::RenderPass {
        self.vk_render_pass
    }
}
