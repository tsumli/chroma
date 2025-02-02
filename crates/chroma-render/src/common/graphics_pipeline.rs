use anyhow::Result;
use ash::vk;

pub struct GraphicsPipeline {
    pipelines: Vec<vk::Pipeline>,
    device: ash::Device,
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        for vk_pipeline in self.pipelines.iter() {
            unsafe {
                self.device.destroy_pipeline(vk_pipeline.clone(), None);
            }
        }
    }
}

impl GraphicsPipeline {
    pub fn new(
        pipeline_create_infos: &[vk::GraphicsPipelineCreateInfo],
        device: ash::Device,
    ) -> Result<Self> {
        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), pipeline_create_infos, None)
                .unwrap()
        };
        Ok(Self { pipelines, device })
    }

    pub fn vk_pipelines(&self) -> &[vk::Pipeline] {
        &self.pipelines
    }
}
