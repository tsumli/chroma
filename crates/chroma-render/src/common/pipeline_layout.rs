use anyhow::Result;
use ash::vk;

#[derive(Clone)]
pub struct PipelineLayout {
    pipeline_layout: vk::PipelineLayout,
    device: ash::Device,
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

impl PipelineLayout {
    pub fn new(
        pipeline_layout_create_info: vk::PipelineLayoutCreateInfo,
        device: ash::Device,
    ) -> Result<Self> {
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None)? };
        Ok(Self {
            pipeline_layout,
            device,
        })
    }

    pub fn vk_pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}
