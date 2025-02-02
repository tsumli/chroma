use anyhow::Result;

fn main() -> Result<()> {
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();
    chroma_scene::asset::init()?;
    Ok(())
}
