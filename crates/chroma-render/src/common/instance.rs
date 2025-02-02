use super::{
    debug::create_debug_messenger_create_info,
    layer::required_layer_names,
};
use crate::common::extension::required_extension_names;
use anyhow::Result;
use ash::vk::{
    self,
    ValidationFeatureEnableEXT,
};
use std::ffi::{
    c_char,
    CString,
};
pub struct Instance {
    instance: ash::Instance,
}

impl Instance {
    pub fn new(entry: &ash::Entry) -> Result<Self> {
        let app_name = CString::new("Chroma")?;
        let engine_name = CString::new("Chroma Engine")?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .api_version(ash::vk::API_VERSION_1_3)
            .engine_name(&engine_name)
            .engine_version(ash::vk::make_api_version(0, 0, 1, 0))
            .application_version(ash::vk::make_api_version(0, 0, 1, 0));

        let extension_names = required_extension_names();

        // enables the required validation layers
        let layer_names_c_char: Vec<CString> = required_layer_names()
            .iter()
            .map(|&name| CString::new(name).unwrap())
            .collect();
        let layer_names_ptrs: Vec<*const c_char> = layer_names_c_char
            .iter()
            .map(|name| name.as_ptr())
            .collect();

        let validation_features =
            vk::ValidationFeaturesEXT::default().enabled_validation_features(&[
                ValidationFeatureEnableEXT::BEST_PRACTICES,
                ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
                // ValidationFeatureEnableEXT::DEBUG_PRINTF, // severely impacts performance
            ]);

        let mut debug_messenger_create_info = create_debug_messenger_create_info();
        debug_messenger_create_info.p_next =
            &validation_features as *const _ as *const std::ffi::c_void;

        // make instance create info and create instance
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names_ptrs)
            .push_next(&mut debug_messenger_create_info);

        let instance = unsafe { entry.create_instance(&create_info, None)? };
        Ok(Self { instance })
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
