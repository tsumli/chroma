use ash::vk;

pub struct DescriptorPool {
    pool: vk::DescriptorPool,
    device: ash::Device,
}

impl DescriptorPool {
    pub fn new(create_info: vk::DescriptorPoolCreateInfo, device: ash::Device) -> Self {
        let pool = unsafe { device.create_descriptor_pool(&create_info, None) }
            .expect("Failed to create descriptor pool");

        Self { pool, device }
    }

    pub fn vk_pool(&self) -> vk::DescriptorPool {
        self.pool
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}
