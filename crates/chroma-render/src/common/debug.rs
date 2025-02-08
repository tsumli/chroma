use anyhow::Result;
use ash::vk::{
    self,
};
use std::ffi::{
    c_void,
    CStr,
};

#[allow(dead_code)]
pub fn set_debug_name<T: ash::vk::Handle>(
    device: ash::ext::debug_utils::Device,
    handle: T,
    name: &str,
) {
    let marker_info = vk::DebugUtilsObjectNameInfoEXT::default()
        .object_name(CStr::from_bytes_with_nul(name.as_bytes()).unwrap())
        .object_handle(handle);
    unsafe {
        device.set_debug_utils_object_name(&marker_info).unwrap();
    }
}

/// the callback function used in Debug Utils.
unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            log::debug!("{} {:?}", types, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!("{} {:?}", types, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            log::error!("{} {:?}", types, message)
        }
        _ => log::info!("{} {:?}", types, message),
    }

    vk::FALSE
}

pub fn create_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(vulkan_debug_utils_callback))
}

fn setup_debug_utils(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> Result<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
    let debug_utils_loader = ash::ext::debug_utils::Instance::new(entry, instance);
    let create_info = create_debug_messenger_create_info();
    let utils_messenger =
        unsafe { debug_utils_loader.create_debug_utils_messenger(&create_info, None)? };
    Ok((debug_utils_loader, utils_messenger))
}

#[derive(Default)]
pub struct DebugUtils {
    debug_utils_loader: Option<ash::ext::debug_utils::Instance>,
    utils_messenger: vk::DebugUtilsMessengerEXT,
}

impl Drop for DebugUtils {
    fn drop(&mut self) {
        unsafe {
            if let Some(instance) = &self.debug_utils_loader {
                instance.destroy_debug_utils_messenger(self.utils_messenger, None);
            }
        }
    }
}

impl DebugUtils {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance) -> Result<Self> {
        let (debug_utils_loader, utils_messenger) = setup_debug_utils(entry, instance)?;
        Ok(Self {
            debug_utils_loader: Some(debug_utils_loader),
            utils_messenger,
        })
    }
}
