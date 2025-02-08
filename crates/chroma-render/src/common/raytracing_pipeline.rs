use ash::vk::{
    self,
    DeferredOperationKHR,
    PipelineCache,
};

#[derive(Clone)]
pub struct RaytracingPipeline {
    pipeline: vk::Pipeline,
    device: ash::Device,
}

impl RaytracingPipeline {
    pub fn new(
        pipeline_info: &vk::RayTracingPipelineCreateInfoKHR,
        logical_device: ash::Device,
        raytracing_pipeline_device: ash::khr::ray_tracing_pipeline::Device,
    ) -> Self {
        let pipeline = unsafe {
            raytracing_pipeline_device
                .create_ray_tracing_pipelines(
                    DeferredOperationKHR::null(),
                    PipelineCache::null(),
                    &[pipeline_info.clone()],
                    None,
                )
                .unwrap()[0]
        };

        Self {
            pipeline,
            device: logical_device,
        }
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
}

impl Drop for RaytracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}
