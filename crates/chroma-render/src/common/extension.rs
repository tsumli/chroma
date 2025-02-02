use ash::{
    ext::debug_utils,
    khr::surface,
};

pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        surface::NAME.as_ptr(),
        ash::khr::xlib_surface::NAME.as_ptr(),
        debug_utils::NAME.as_ptr(),
        ash::khr::get_physical_device_properties2::NAME.as_ptr(),
    ]
}
