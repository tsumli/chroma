#[derive(Clone)]
pub struct ExtraFunctions {
    pub acceleration_structure: ash::khr::acceleration_structure::Device,
    pub raytracing_pipeline: ash::khr::ray_tracing_pipeline::Device,
    pub mesh_shader: ash::ext::mesh_shader::Device,
    pub get_physical_device_properties2: ash::khr::get_physical_device_properties2::Instance,

    #[allow(dead_code)]
    pub debug_utils: ash::ext::debug_utils::Device,
}

impl ExtraFunctions {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance, device: &ash::Device) -> Self {
        let acceleration_structure =
            ash::khr::acceleration_structure::Device::new(instance, device);
        let raytracing_pipeline = ash::khr::ray_tracing_pipeline::Device::new(instance, device);
        let mesh_shader = ash::ext::mesh_shader::Device::new(instance, device);
        let get_physical_device_properties2 =
            ash::khr::get_physical_device_properties2::Instance::new(entry, instance);
        let debug_utils = ash::ext::debug_utils::Device::new(instance, device);
        Self {
            acceleration_structure,
            raytracing_pipeline,
            mesh_shader,
            get_physical_device_properties2,
            debug_utils,
        }
    }
}
