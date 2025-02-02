use anyhow::Result;
use ash::{
    prelude::VkResult,
    vk,
};
use std::{
    fs::File,
    io::Read,
    path::Path,
};

pub fn read_shader_code(shader_path: &Path) -> Result<Vec<u8>> {
    anyhow::ensure!(
        shader_path.exists(),
        "Shader path does not exist: {:?}",
        shader_path
    );
    let spv_file = File::open(shader_path)?;
    let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();
    Ok(bytes_code)
}

pub fn create_shader_module(device: &ash::Device, code: Vec<u8>) -> VkResult<vk::ShaderModule> {
    let code_aligned = unsafe {
        std::slice::from_raw_parts(
            code.as_ptr() as *const u32,
            code.len() / std::mem::size_of::<u32>(),
        )
    };
    let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(code_aligned);

    unsafe { device.create_shader_module(&shader_module_create_info, None) }
}
