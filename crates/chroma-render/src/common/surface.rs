use anyhow::{
    bail,
    Context,
    Result,
};
use ash::vk;
use winit::raw_window_handle::{
    HasDisplayHandle,
    HasWindowHandle,
    RawDisplayHandle,
    RawWindowHandle,
};

pub struct Surface {
    pub surface: vk::SurfaceKHR,
    pub surface_loader: ash::khr::surface::Instance,
}

impl Surface {
    pub fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &winit::window::Window,
    ) -> Result<Self> {
        let surface = unsafe { create_surface_platform(entry, instance, window)? };
        let surface_loader = ash::khr::surface::Instance::new(entry, instance);

        Ok(Self {
            surface,
            surface_loader,
        })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

unsafe fn create_surface_platform(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR> {
    log::info!("create_surface_platform");
    let raw_window = window.window_handle()?.as_raw();
    let xlib_window = match raw_window {
        RawWindowHandle::Xlib(handle) => handle.window,
        _ => bail!("Expected Xlib window handle"),
    };
    let raw_display = window.display_handle()?.as_raw();
    let xlib_display = match raw_display {
        RawDisplayHandle::Xlib(handle) => handle.display.context("Expected Xlib display handle")?,
        _ => bail!("Expected Xlib display handle"),
    };

    let surface_create_info = vk::XlibSurfaceCreateInfoKHR::default()
        .window(xlib_window)
        .dpy(xlib_display.as_ptr() as *mut std::ffi::c_void);

    let xlib_surface = ash::khr::xlib_surface::Instance::new(entry, instance);
    let surface = unsafe { xlib_surface.create_xlib_surface(&surface_create_info, None)? };
    Ok(surface)
}
