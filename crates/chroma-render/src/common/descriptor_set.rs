use ash::vk;

#[derive(Clone)]
pub struct DescriptorSet {
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    device: ash::Device,
}

impl DescriptorSet {
    pub fn new(
        allocate_info: vk::DescriptorSetAllocateInfo,
        descriptor_pool: vk::DescriptorPool,
        device: ash::Device,
    ) -> Self {
        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&allocate_info)
                .expect("Failed to allocate descriptor set")
        };

        Self {
            descriptor_sets,
            descriptor_pool,
            device,
        }
    }

    pub fn vk_descriptor_set(&self, index: usize) -> vk::DescriptorSet {
        self.descriptor_sets[index]
    }

    pub fn vk_descriptor_sets(&self) -> &Vec<vk::DescriptorSet> {
        &self.descriptor_sets
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        unsafe {
            self.device
                .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets)
                .unwrap();
        }
    }
}
