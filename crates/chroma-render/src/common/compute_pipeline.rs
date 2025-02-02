use anyhow::Result;
use ash::vk;

#[derive(Clone)]
pub struct ComputePipeline {
    pipelines: Vec<vk::Pipeline>,
    device: ash::Device,
}

impl ComputePipeline {
    pub fn new(
        pipeline_create_infos: &[vk::ComputePipelineCreateInfo],
        device: ash::Device,
    ) -> Result<Self> {
        let pipelines = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), pipeline_create_infos, None)
                .unwrap()
        };
        Ok(Self { pipelines, device })
    }

    pub fn vk_pipelines(&self) -> &[vk::Pipeline] {
        &self.pipelines
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        for vk_pipeline in self.pipelines.iter() {
            unsafe {
                self.device.destroy_pipeline(vk_pipeline.clone(), None);
            }
        }
    }
}
