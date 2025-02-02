use anyhow::Result;
use std::{
    ffi::CStr,
    os::raw::c_char,
};

/// Helper function to convert [c_char; SIZE] to string
pub fn convert_char_to_string(raw_string_array: &[c_char]) -> Result<String> {
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    Ok(raw_string.to_str()?.to_owned())
}
