use anyhow::Result;
use winit::{
    dpi::PhysicalSize,
    event_loop::ActiveEventLoop,
    window::WindowAttributes,
};

pub fn create_window(
    event_loop: &ActiveEventLoop,
    window_size: PhysicalSize<u32>,
) -> Result<winit::window::Window> {
    let window_attributes = WindowAttributes::default()
        .with_title("Chroma")
        .with_inner_size(window_size);
    Ok(event_loop.create_window(window_attributes)?)
}
