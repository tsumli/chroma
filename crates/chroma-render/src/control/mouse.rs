use std::collections::HashMap;
use winit::{
    dpi::PhysicalPosition,
    event::{
        ElementState,
        MouseButton,
    },
};

#[derive(Debug, Default)]
pub struct Mouse {
    position: PhysicalPosition<f32>,
    is_pressed: HashMap<MouseButton, bool>,
    is_toggled: HashMap<MouseButton, bool>,
}

impl Mouse {
    pub fn set_position(&mut self, position: PhysicalPosition<f32>) {
        self.position = position;
    }

    pub fn position(&self) -> PhysicalPosition<f32> {
        self.position
    }

    pub fn position_from_center(&self, center: PhysicalPosition<f32>) -> PhysicalPosition<f32> {
        PhysicalPosition::<f32>::new(self.position.x - center.x, self.position.y - center.y)
    }

    pub fn set_input(&mut self, state: ElementState, button: MouseButton) {
        match state {
            ElementState::Pressed => {
                self.is_pressed.insert(button, true);
            }
            ElementState::Released => {
                self.is_pressed.insert(button, false);

                // Toggle the state of the button.
                let toggled = self.is_toggled.entry(button).or_insert(false);
                *toggled = !*toggled;
            }
        }
    }

    #[allow(dead_code)]
    pub fn is_pressed(&self, button: &MouseButton) -> bool {
        self.is_pressed.get(button).copied().unwrap_or_default()
    }

    pub fn is_toggled(&self, button: &MouseButton) -> bool {
        self.is_toggled.get(button).copied().unwrap_or_default()
    }
}
