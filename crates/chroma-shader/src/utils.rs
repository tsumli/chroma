use anyhow::{
    Context,
    Result,
};
use glob::glob;
use std::{
    collections::HashSet,
    path::{
        Path,
        PathBuf,
    },
};

pub fn glob_shader_src(shader_src_root: &Path, extensions: &HashSet<&str>) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in glob(
        shader_src_root
            .join("**/*")
            .to_str()
            .context("failed to convert to str")?,
    )? {
        match entry {
            Ok(path) => {
                if let Some(ext) = path.extension() {
                    if extensions.contains(&ext.to_str().context("failed to convert to str")?) {
                        paths.push(path);
                    }
                }
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok(paths)
}
