use crate::common::consts::MAX_FRAMES_IN_FLIGHT;
use anyhow::Result;
use ash::vk;

pub struct SyncObjects {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    inflight_fences: Vec<vk::Fence>,
    device: ash::Device,
}

impl SyncObjects {
    pub fn new(device: ash::Device) -> Result<Self> {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = vec![];
        let mut render_finished_semaphores = vec![];
        let mut inflight_fences = vec![];

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                let image_available_semaphore =
                    device.create_semaphore(&semaphore_create_info, None)?;
                let render_finished_semaphore =
                    device.create_semaphore(&semaphore_create_info, None)?;
                let inflight_fence = device.create_fence(&fence_create_info, None)?;

                image_available_semaphores.push(image_available_semaphore);
                render_finished_semaphores.push(render_finished_semaphore);
                inflight_fences.push(inflight_fence);
            }
        }

        // reset fence
        unsafe {
            device.reset_fences(&inflight_fences).unwrap();
        }

        Ok(Self {
            image_available_semaphores,
            render_finished_semaphores,
            inflight_fences,
            device,
        })
    }

    pub fn image_available_semaphores(&self) -> &Vec<vk::Semaphore> {
        &self.image_available_semaphores
    }

    pub fn render_finished_semaphores(&self) -> &Vec<vk::Semaphore> {
        &self.render_finished_semaphores
    }

    pub fn inflight_fences(&self) -> &Vec<vk::Fence> {
        &self.inflight_fences
    }
}

impl Drop for SyncObjects {
    fn drop(&mut self) {
        unsafe {
            for &semaphore in self.image_available_semaphores.iter() {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in self.render_finished_semaphores.iter() {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &fence in self.inflight_fences.iter() {
                self.device.destroy_fence(fence, None);
            }
        }
    }
}
