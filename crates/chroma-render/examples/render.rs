use chroma_base::path::get_project_root;
use chroma_render::app::app::App;
use chroma_scene::scene::Scene;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Configs {
    scene: String,
    width: u32,
    height: u32,
}

fn main() -> anyhow::Result<()> {
    std::env::set_var("RUST_LOG", "debug");
    env_logger::init();

    log::info!("Parsing arguments");
    let config_path = get_project_root()?.join("crates/chroma-render/configs/render.json");
    let configs: Configs = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

    log::info!("Building assets");
    chroma_scene::asset::init()?;

    log::info!("Compiling shaders");
    chroma_shader::command::compile_all()?;

    log::info!("Starting Chroma Render");
    let event_loop = winit::event_loop::EventLoop::builder().build()?;
    let mut app = App::default();
    app.set_window_size(winit::dpi::PhysicalSize::new(configs.width, configs.height));
    app.set_scene(Scene::from_scene_name(&configs.scene)?);
    event_loop.run_app(&mut app)?;
    Ok(())
}
