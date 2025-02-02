use ash::vk;

pub struct DescriptorSetLayout {
    descriptor_set_layout: vk::DescriptorSetLayout,
    device: ash::Device,
}

impl DescriptorSetLayout {
    pub fn new(create_info: vk::DescriptorSetLayoutCreateInfo, device: ash::Device) -> Self {
        let pool = unsafe { device.create_descriptor_set_layout(&create_info, None) }
            .expect("Failed to create descriptor pool");

        Self {
            descriptor_set_layout: pool,
            device,
        }
    }

    pub fn vk_descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
