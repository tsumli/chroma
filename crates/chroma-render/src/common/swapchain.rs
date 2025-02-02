use super::{
    device::QueueFamilyIndices,
    image_view::ImageView,
    surface::Surface,
};
use crate::common::consts::MAX_FRAMES_IN_FLIGHT;
use anyhow::Result;
use ash::vk;
use winit::dpi::PhysicalSize;

pub struct Swapchain {
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<ImageView>,
}

impl Swapchain {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
        queue_family: &QueueFamilyIndices,
        size: PhysicalSize<u32>,
    ) -> Result<Self> {
        let swapchain_support = query_swapchain_support(physical_device, surface)?;
        let surface_format = choose_swapchain_format(&swapchain_support.formats);
        let present_mode = choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = choose_swapchain_extent(&swapchain_support.capabilities, size);

        let (image_sharing_mode, _queue_family_index_count, queue_family_indices) =
            if queue_family.graphics_compute_family != queue_family.present_family {
                (
                    vk::SharingMode::CONCURRENT,
                    2,
                    vec![
                        queue_family.graphics_compute_family.unwrap(),
                        queue_family.present_family.unwrap(),
                    ],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, vec![])
            };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.surface)
            .min_image_count(MAX_FRAMES_IN_FLIGHT as u32)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(queue_family_indices.as_slice())
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        // create image views for all swapchain images
        let mut swapchain_image_views = vec![];
        for &image in swapchain_images.iter() {
            let imageview_create_info = vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image(image);

            let imageview = ImageView::new(imageview_create_info, device.clone())?;
            swapchain_image_views.push(imageview);
        }
        Ok(Swapchain {
            swapchain_loader,
            swapchain,
            swapchain_format: surface_format.format,
            swapchain_extent: extent,
            swapchain_images,
            swapchain_image_views,
        })
    }

    pub fn swapchain_loader(&self) -> &ash::khr::swapchain::Device {
        &self.swapchain_loader
    }

    pub fn vk_swapchain(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub fn vk_swapchain_images(&self) -> &Vec<vk::Image> {
        &self.swapchain_images
    }

    pub fn vk_swapchain_format(&self) -> vk::Format {
        self.swapchain_format
    }

    pub fn vk_swapchain_extent(&self) -> vk::Extent2D {
        self.swapchain_extent
    }

    #[allow(dead_code)]
    pub fn vk_swapchain_image_views(&self) -> Vec<vk::ImageView> {
        let mut vk_image_views = vec![];
        for image_view in &self.swapchain_image_views {
            vk_image_views.push(image_view.vk_image_view());
        }
        vk_image_views
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

pub struct SwapChainSupportDetail {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

fn choose_swapchain_format(available_formats: &Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
    // check if list contains most widely used R8G8B8A8 format with nonlinear color space
    for available_format in available_formats {
        if available_format.format == vk::Format::B8G8R8A8_UNORM
            && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        {
            return available_format.clone();
        }
    }

    // return the first format from the list
    return available_formats.first().unwrap().clone();
}

fn choose_swapchain_present_mode(
    available_present_modes: &Vec<vk::PresentModeKHR>,
) -> vk::PresentModeKHR {
    for &available_present_mode in available_present_modes.iter() {
        if available_present_mode == vk::PresentModeKHR::MAILBOX {
            return available_present_mode;
        }
    }
    vk::PresentModeKHR::FIFO
}

fn choose_swapchain_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    size: PhysicalSize<u32>,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::max_value() {
        capabilities.current_extent
    } else {
        vk::Extent2D::default()
            .width(size.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ))
            .height(size.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ))
    }
}

pub fn query_swapchain_support(
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
) -> Result<SwapChainSupportDetail> {
    unsafe {
        let capabilities = surface
            .surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface.surface)?;
        let formats = surface
            .surface_loader
            .get_physical_device_surface_formats(physical_device, surface.surface)?;
        let present_modes = surface
            .surface_loader
            .get_physical_device_surface_present_modes(physical_device, surface.surface)?;

        Ok(SwapChainSupportDetail {
            capabilities,
            formats,
            present_modes,
        })
    }
}
