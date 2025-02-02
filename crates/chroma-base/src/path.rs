use anyhow::Result;
use git2;
use std::path::PathBuf;

pub fn get_project_root() -> Result<PathBuf> {
    let repo = git2::Repository::discover(std::env::current_dir()?)?;
    let workdir = repo
        .workdir()
        .ok_or_else(|| git2::Error::from_str("No workdir"))?;
    Ok(workdir.to_path_buf())
}

pub fn get_shader_root() -> Result<PathBuf> {
    let project_root = get_project_root()?;
    Ok(project_root.join("shader"))
}

pub fn get_shader_src_root() -> Result<PathBuf> {
    let shader_root = get_shader_root()?;
    Ok(shader_root.join("src"))
}

pub fn get_shader_spv_root() -> Result<PathBuf> {
    let shader_root = get_shader_root()?;
    Ok(shader_root.join("spv"))
}
