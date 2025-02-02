use anyhow::{
    bail,
    Result,
};
use ash::vk;

pub fn find_memory_type(
    type_filter: u32,
    required_properties: vk::MemoryPropertyFlags,
    physical_device: vk::PhysicalDevice,
    instance: ash::Instance,
) -> Result<u32> {
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    for (type_i, mem_type) in mem_props.memory_types.iter().enumerate() {
        if type_filter & (1 << type_i) != 0 && mem_type.property_flags.contains(required_properties)
        {
            return Ok(type_i as u32);
        }
    }
    bail!(
        "failed to find suitable memory type! type_filter: {:?}, properties: {:?}",
        type_filter,
        required_properties
    );
}
