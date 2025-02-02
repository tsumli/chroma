use anyhow::{
    ensure,
    Result,
};
use chroma_base::path::get_project_root;
use chroma_scene::scene::Scene;
use serde::Deserialize;

fn main() -> Result<()> {
    std::env::set_var("RUST_LOG", "debug");
    env_logger::init();

    let config_path = get_project_root()?
        .join("crates")
        .join("chroma-scene")
        .join("config")
        .join("dragon_attenuation.json");
    ensure!(
        config_path.exists(),
        "Config doesn't exist: {:?}",
        config_path
    );
    let mut deserializer = serde_json::Deserializer::from_reader(std::fs::File::open(config_path)?);
    let scene_descriptor = chroma_scene::scene::SceneDescriptor::deserialize(&mut deserializer)?;
    log::info!("{:#?}", scene_descriptor);

    let _scene = Scene::from_scene_descriptor(scene_descriptor)?;
    Ok(())
}
