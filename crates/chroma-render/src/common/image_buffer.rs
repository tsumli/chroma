use super::memory::find_memory_type;
use anyhow::Result;
use ash::vk::{
    self,
    MemoryAllocateInfo,
};

#[derive(Clone)]
pub struct ImageBuffer {
    image: vk::Image,
    image_view: vk::ImageView,
    device: ash::Device,
    format: vk::Format,
    extent: vk::Extent3D,
    device_memory: vk::DeviceMemory,
}

impl ImageBuffer {
    pub fn new(
        image_create_info: &vk::ImageCreateInfo,
        view_type: vk::ImageViewType,
        memory_property_flags: vk::MemoryPropertyFlags,
        aspect_mask: vk::ImageAspectFlags,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        instance: ash::Instance,
    ) -> Result<Self> {
        let image = unsafe { device.create_image(&image_create_info, None).unwrap() };

        // allocate image memory
        let memory_requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_type_index = find_memory_type(
            memory_requirements.memory_type_bits,
            memory_property_flags,
            physical_device,
            instance.clone(),
        )?;
        let device_memory = unsafe {
            device.allocate_memory(
                &MemoryAllocateInfo::default()
                    .allocation_size(memory_requirements.size)
                    .memory_type_index(memory_type_index),
                None,
            )?
        };

        // bind image memory
        let bind_infos = [vk::BindImageMemoryInfo::default()
            .image(image)
            .memory(device_memory)
            .memory_offset(0)];
        unsafe {
            device.bind_image_memory2(&bind_infos)?;
        };

        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .flags(vk::ImageViewCreateFlags::empty())
            .format(image_create_info.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .base_mip_level(0)
                    .level_count(image_create_info.mip_levels)
                    .base_array_layer(0)
                    .layer_count(image_create_info.array_layers),
            );

        let image_view = unsafe {
            device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        };

        Ok(Self {
            image,
            image_view,
            device,
            format: image_create_info.format,
            extent: image_create_info.extent,
            device_memory,
        })
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn extent(&self) -> vk::Extent3D {
        self.extent
    }

    pub fn image_view(&self) -> vk::ImageView {
        self.image_view
    }
}

impl Drop for ImageBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.image_view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.device_memory, None);
        }
    }
}
