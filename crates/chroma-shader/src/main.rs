use anyhow::Result;
use chroma_shader::command::compile_all;

fn main() -> Result<()> {
    std::env::set_var("RUST_LOG", "debug");
    env_logger::init();
    compile_all()?;
    Ok(())
}
