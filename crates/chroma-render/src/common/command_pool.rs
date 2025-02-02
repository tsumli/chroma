use anyhow::Result;
use ash::vk;

pub struct CommandPool {
    command_pool: vk::CommandPool,
    device: ash::Device,
}

impl CommandPool {
    pub fn new(
        command_pool_create_info: &vk::CommandPoolCreateInfo,
        device: ash::Device,
    ) -> Result<Self> {
        let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None)? };

        Ok(Self {
            command_pool,
            device,
        })
    }

    pub fn vk_command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
