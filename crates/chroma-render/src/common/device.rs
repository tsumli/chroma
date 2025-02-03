use super::{
    surface::Surface,
    swapchain::query_swapchain_support,
};
use crate::utils::tool::convert_char_to_string;
use anyhow::{
    bail,
    Result,
};
use ash::vk;
use std::{
    collections::HashSet,
    ffi::CStr,
};

const DEVICE_EXTENSIONS: [&CStr; 7] = [
    ash::khr::swapchain::NAME,
    ash::ext::robustness2::NAME,
    ash::khr::deferred_host_operations::NAME,
    ash::khr::pipeline_library::NAME,
    ash::khr::ray_tracing_pipeline::NAME,
    ash::khr::acceleration_structure::NAME,
    ash::ext::mesh_shader::NAME,
];

pub fn pick_physical_device(
    instance: &ash::Instance,
    surface: &Surface,
) -> Result<vk::PhysicalDevice> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    log::debug!(
        "{} devices (GPU) found with vulkan support.",
        physical_devices.len()
    );

    let mut result = None;
    for &physical_device in physical_devices.iter() {
        if let Ok(_) = is_physical_device_suitable(instance, physical_device, surface) {
            if result.is_none() {
                result = Some(physical_device)
            }
        }
    }

    match result {
        None => bail!("Failed to find a suitable GPU!"),
        Some(physical_device) => Ok(physical_device),
    }
}

pub fn find_queue_family(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: &super::surface::Surface,
) -> Result<QueueFamilyIndices> {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let mut queue_family_indices = QueueFamilyIndices::new();

    let mut index = 0;
    for queue_family in queue_families.iter() {
        if queue_family.queue_count > 0
            && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE)
        {
            queue_family_indices.graphics_compute_family = Some(index);
        }

        let is_present_support = unsafe {
            surface.surface_loader.get_physical_device_surface_support(
                physical_device,
                index as u32,
                surface.surface,
            )?
        };
        if queue_family.queue_count > 0 && is_present_support {
            queue_family_indices.present_family = Some(index);
        }

        if queue_family_indices.is_complete() {
            break;
        }

        index += 1;
    }

    Ok(queue_family_indices)
}

fn is_physical_device_suitable(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_stuff: &Surface,
) -> Result<bool> {
    let _device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let _device_features = unsafe { instance.get_physical_device_features(physical_device) };

    let indices = find_queue_family(instance, physical_device, surface_stuff)?;
    let is_queue_family_supported = indices.is_complete();
    let is_device_extension_supported = check_device_extension_support(instance, physical_device)?;

    let is_swapchain_supported = if is_device_extension_supported {
        let swapchain_support = query_swapchain_support(physical_device, surface_stuff)?;
        !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
    } else {
        false
    };

    Ok(is_queue_family_supported && is_device_extension_supported && is_swapchain_supported)
}

pub struct QueueFamilyIndices {
    pub graphics_compute_family: Option<u32>,
    pub present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn new() -> QueueFamilyIndices {
        QueueFamilyIndices {
            graphics_compute_family: None,
            present_family: None,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.graphics_compute_family.is_some() && self.present_family.is_some()
    }
}

pub fn create_logical_device(
    indices: &QueueFamilyIndices,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<Device> {
    let mut unique_queue_families = std::collections::HashSet::new();
    unique_queue_families.insert(indices.graphics_compute_family.unwrap());
    unique_queue_families.insert(indices.present_family.unwrap());

    let queue_priorities = [1.0_f32];
    let mut queue_create_infos = vec![];
    for &queue_family in unique_queue_families.iter() {
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities);
        queue_create_infos.push(queue_create_info);
    }

    let mut physical_device_maintenance4_features =
        vk::PhysicalDeviceMaintenance4FeaturesKHR::default().maintenance4(true);

    let mut physical_device_eight_bit_storage_features =
        vk::PhysicalDevice8BitStorageFeatures::default()
            .storage_buffer8_bit_access(true)
            .uniform_and_storage_buffer8_bit_access(true);
    physical_device_eight_bit_storage_features.p_next =
        &mut physical_device_maintenance4_features as *mut _ as *mut std::ffi::c_void;

    let mut physical_device_mesh_shader_features =
        vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
            .task_shader(true)
            .mesh_shader(true);
    physical_device_mesh_shader_features.p_next =
        &mut physical_device_eight_bit_storage_features as *mut _ as *mut std::ffi::c_void;

    // physical_device_timeline_semaphore_features
    let mut physical_device_timeline_semaphore_features =
        vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
    physical_device_timeline_semaphore_features.p_next =
        &mut physical_device_mesh_shader_features as *mut _ as *mut std::ffi::c_void;

    // raytracing_pipeline_features
    let mut raytracing_pipeline_features =
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
            .ray_tracing_pipeline(true)
            .ray_tracing_pipeline_trace_rays_indirect(true)
            .ray_traversal_primitive_culling(true);
    raytracing_pipeline_features.p_next =
        &mut physical_device_timeline_semaphore_features as *mut _ as *mut std::ffi::c_void;

    // acceleration_structure_features
    let mut acceleration_structure_features =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true)
            .acceleration_structure_capture_replay(true)
            .descriptor_binding_acceleration_structure_update_after_bind(true);
    acceleration_structure_features.p_next =
        &mut raytracing_pipeline_features as *mut _ as *mut std::ffi::c_void;

    // scalar_block_layout_features
    let mut scalar_block_layout_features =
        vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT::default().scalar_block_layout(true);
    scalar_block_layout_features.p_next =
        &mut acceleration_structure_features as *mut _ as *mut std::ffi::c_void;

    // synchronization2_features
    let mut synchronization2_features =
        vk::PhysicalDeviceSynchronization2FeaturesKHR::default().synchronization2(true);
    synchronization2_features.p_next =
        &mut scalar_block_layout_features as *mut _ as *mut std::ffi::c_void;

    // robustness2_features
    let mut robustness2_features = vk::PhysicalDeviceRobustness2FeaturesEXT::default()
        .null_descriptor(true)
        .robust_buffer_access2(true)
        .robust_image_access2(true);
    robustness2_features.p_next = &mut synchronization2_features as *mut _ as *mut std::ffi::c_void;

    // buffer_device_address_features
    let mut buffer_device_address_features =
        vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
            .buffer_device_address(true)
            .buffer_device_address_capture_replay(true)
            .buffer_device_address_multi_device(true);
    buffer_device_address_features.p_next =
        &mut robustness2_features as *mut _ as *mut std::ffi::c_void;

    // physical_device_features
    let mut physical_device_features = vk::PhysicalDeviceFeatures2::default()
        .features(
            vk::PhysicalDeviceFeatures::default()
                .shader_int64(true)
                .robust_buffer_access(true)
                .sampler_anisotropy(true)
                .shader_clip_distance(true),
        )
        .push_next(&mut buffer_device_address_features);

    let extension_names = DEVICE_EXTENSIONS
        .iter()
        .map(|extension| extension.as_ptr())
        .collect::<Vec<_>>();
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&extension_names)
        .push_next(&mut physical_device_features);
    let device = Device::new(&device_create_info, physical_device, instance)?;
    Ok(device)
}

fn check_device_extension_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<bool> {
    let available_extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .expect("Failed to get device extension properties.")
    };

    let mut available_extension_names = vec![];

    for extension in available_extensions.iter() {
        let extension_name = convert_char_to_string(&extension.extension_name)?;
        available_extension_names.push(extension_name);
    }

    let mut required_extensions: HashSet<String> = HashSet::new();
    for extension in DEVICE_EXTENSIONS.iter() {
        required_extensions.insert(extension.to_string_lossy().to_string());
    }

    for extension_name in available_extension_names.iter() {
        required_extensions.remove(extension_name);
    }

    Ok(required_extensions.is_empty())
}

#[derive(Clone)]
pub struct Device {
    device: ash::Device,
}

impl Device {
    pub fn new(
        device_create_info: &vk::DeviceCreateInfo,
        physical_device: vk::PhysicalDevice,
        instance: &ash::Instance,
    ) -> Result<Self> {
        let device = unsafe { instance.create_device(physical_device, device_create_info, None)? };
        Ok(Self { device })
    }

    pub fn ash_device(&self) -> &ash::Device {
        &self.device
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
