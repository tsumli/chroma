use super::window::create_window;
use crate::control::Control;
use imgui_winit_support::WinitPlatform;
use winit::{
    application::ApplicationHandler,
    dpi::{
        PhysicalPosition,
        PhysicalSize,
    },
    event::{
        Event,
        WindowEvent,
    },
    event_loop::ActiveEventLoop,
    window::{
        Window,
        WindowId,
    },
};

#[derive(Default)]
pub struct App {
    window: Option<Window>,
    window_size: PhysicalSize<u32>,
    renderer: Option<super::renderer::Renderer>,
    imgui_platform: Option<WinitPlatform>,
    imgui_context: Option<imgui::Context>,
    control: Control,
    scene: chroma_scene::scene::Scene,
}

impl App {
    pub fn set_scene(&mut self, scene: chroma_scene::scene::Scene) {
        self.scene = scene
    }

    pub fn set_window_size(&mut self, size: PhysicalSize<u32>) {
        self.window_size = size;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // setup window
        let window = create_window(event_loop, self.window_size).expect("failed to create window");
        self.window = Some(window);

        // setup imgui
        let mut imgui_context = imgui::Context::create();
        imgui_context.set_ini_filename(None);
        let imgui_platform = WinitPlatform::new(&mut imgui_context);
        self.imgui_platform = Some(imgui_platform);
        self.imgui_context = Some(imgui_context);

        // setup renderer
        self.renderer = Some(
            super::renderer::Renderer::new(
                self.window.as_ref().unwrap(),
                self.window_size,
                &self.scene,
                &mut self.imgui_platform.as_mut().unwrap(),
                &mut self.imgui_context.as_mut().unwrap(),
            )
            .unwrap(),
        );

        // setup control
        self.control = Control::default();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        window_event: WindowEvent,
    ) {
        let renderer = self.renderer.as_mut().unwrap();
        let window = self.window.as_ref().unwrap();
        let imgui_platform = self.imgui_platform.as_mut().unwrap();
        let imgui_context = self.imgui_context.as_mut().unwrap();

        let generic_event: Event<()> = Event::WindowEvent {
            window_id,
            event: window_event.clone(),
        };

        imgui_platform.handle_event(imgui_context.io_mut(), window, &generic_event);

        match window_event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                renderer.device_wait_idle();
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.control.keyboard_mut().input_event(&event);
                if self
                    .control
                    .keyboard()
                    .is_pressed(&winit::keyboard::KeyCode::Escape)
                {
                    event_loop.exit();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.control
                    .mouse_mut()
                    .set_position(PhysicalPosition::<f32>::new(
                        position.x as f32,
                        position.y as f32,
                    ));

                // after `set_cursor_position`, `CursorMoved` event is fired. So we need to call
                // `request_redraw` here.
                window.request_redraw();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.control.mouse_mut().set_input(state, button);
            }
            WindowEvent::RedrawRequested => {
                if self
                    .control
                    .mouse()
                    .is_toggled(&winit::event::MouseButton::Right)
                {
                    window
                        .set_cursor_position(winit::dpi::PhysicalPosition::new(
                            self.window_size.width as f64 / 2.0,
                            self.window_size.height as f64 / 2.0,
                        ))
                        .unwrap();
                    window.set_cursor_visible(false);
                } else {
                    window.set_cursor_visible(true);
                }
                renderer.draw_frame(window, imgui_platform, imgui_context, &self.control);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        } else {
            event_loop.exit();
        }
    }
}
