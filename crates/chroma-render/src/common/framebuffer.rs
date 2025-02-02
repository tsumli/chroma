use ash::vk;

#[derive(Clone)]
pub struct Framebuffer {
    framebuffer: vk::Framebuffer,
    device: ash::Device,
}

impl Framebuffer {
    pub fn new(
        render_pass: vk::RenderPass,
        attachments: &[vk::ImageView],
        width: u32,
        height: u32,
        device: ash::Device,
    ) -> Self {
        let framebuffer_create_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(attachments)
            .width(width)
            .height(height)
            .layers(1);

        let framebuffer = unsafe {
            device
                .create_framebuffer(&framebuffer_create_info, None)
                .expect("failed to create Framebuffer!")
        };

        Self {
            framebuffer,
            device,
        }
    }

    pub fn vk_framebuffer(&self) -> &vk::Framebuffer {
        &self.framebuffer
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_framebuffer(self.framebuffer, None);
        }
    }
}
