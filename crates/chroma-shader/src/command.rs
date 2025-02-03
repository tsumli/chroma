use anyhow::{
    ensure,
    Context,
    Result,
};
use chroma_base::path::{
    get_shader_spv_root,
    get_shader_src_root,
};
use log;
use std::{
    io::Write,
    path::Path,
    process::Command,
};

pub fn compile(input_path: &Path) -> Result<()> {
    let input_filename = input_path
        .file_name()
        .context("failed to get file name")?
        .to_str()
        .context("failed to convert to string")?;
    let output_filename = format!("{}.spv", input_filename);

    let shader_src_root = get_shader_src_root()?;
    let relative_input_path = input_path.strip_prefix(&shader_src_root.as_os_str())?;
    let relative_output_path = relative_input_path.with_file_name(output_filename);

    let shader_spv_root = get_shader_spv_root()?;
    let output_path = shader_spv_root.join(relative_output_path);

    if !output_path
        .parent()
        .context("failed to get parent")?
        .exists()
    {
        std::fs::create_dir_all(output_path.parent().context("failed to get parent")?)?;
    }

    log::info!(
        "compiling shader: {} -> {}",
        input_path.display(),
        output_path.display()
    );
    let output = Command::new("glslc")
        .arg(input_path.as_os_str())
        .arg("--target-spv=spv1.6")
        .arg("-g")
        .arg("-O0")
        .arg("-o")
        .arg(output_path.as_os_str())
        .output()?;
    std::io::stderr().write_all(&output.stderr).unwrap();
    ensure!(
        output.status.success(),
        "failed to compile shader: {}",
        input_path.display()
    );
    Ok(())
}

pub fn compile_all() -> Result<()> {
    let shader_src_root = get_shader_src_root().unwrap();
    let extensions = [
        "vert", "frag", "comp", "rgen", "rmiss", "rchit", "rahit", "mesh", "task",
    ]
    .iter()
    .cloned()
    .collect();
    let target_paths = crate::utils::glob_shader_src(&shader_src_root, &extensions).unwrap();
    for path in target_paths {
        compile(&path)?;
    }
    Ok(())
}
