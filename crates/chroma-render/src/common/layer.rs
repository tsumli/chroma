use crate::utils::tool::convert_char_to_string;
use anyhow::Result;

const LAYER_KHRONOS_VALIDATION: &str = "VK_LAYER_KHRONOS_validation";

pub fn required_layer_names() -> Vec<&'static str> {
    vec![LAYER_KHRONOS_VALIDATION]
}

pub fn check_layer_support(entry: &ash::Entry, layer_names: &[&str]) -> Result<bool> {
    let layer_properties = unsafe {
        entry
            .enumerate_instance_layer_properties()
            .expect("Failed to enumerate Instance Layers Properties!")
    };

    if layer_properties.len() <= 0 {
        log::warn!("No available layers.");
        return Ok(false);
    } else {
        log::debug!("Instance Available Layers: ");
        for layer in layer_properties.iter() {
            let layer_name = convert_char_to_string(&layer.layer_name)?;
            log::debug!("\t{}", layer_name);
        }
    }

    for required_layer_name in layer_names.iter() {
        let mut is_layer_found = false;

        for layer_property in layer_properties.iter() {
            let test_layer_name = convert_char_to_string(&layer_property.layer_name)?;
            if (*required_layer_name) == test_layer_name {
                is_layer_found = true;
                break;
            }
        }

        if is_layer_found == false {
            return Ok(false);
        }
    }

    Ok(true)
}
