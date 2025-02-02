use anyhow::Result;
use ash::vk;

pub struct ImageView {
    image_view: vk::ImageView,
    device: ash::Device,
}

impl ImageView {
    pub fn new(
        image_view_create_info: vk::ImageViewCreateInfo,
        device: ash::Device,
    ) -> Result<Self> {
        let image_view = unsafe { device.create_image_view(&image_view_create_info, None)? };
        Ok(Self { image_view, device })
    }

    pub fn vk_image_view(&self) -> vk::ImageView {
        self.image_view
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.image_view, None);
        }
    }
}
