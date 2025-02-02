use ash::vk;

pub fn create_texture_sampler(
    mip_level: u32,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    instance: &ash::Instance,
    entry: &ash::Entry,
) -> vk::Sampler {
    let mut raytracing_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut raytracing_props);
    let physical_device_properties2 =
        ash::khr::get_physical_device_properties2::Instance::new(&entry, &instance);
    {
        unsafe {
            physical_device_properties2.get_physical_device_properties2(physical_device, &mut props)
        }
    }

    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(props.properties.limits.max_sampler_anisotropy)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .max_lod(mip_level as f32);

    let sampler = unsafe { device.create_sampler(&create_info, None).unwrap() };

    sampler
}
