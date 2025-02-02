use std::collections::HashMap;
use winit::{
    event::KeyEvent,
    keyboard::KeyCode,
};

#[derive(Debug, Default)]
pub struct Keyboard {
    is_pressing: HashMap<KeyCode, bool>,
}

impl Keyboard {
    pub fn input_event(&mut self, key: &KeyEvent) {
        if let winit::keyboard::PhysicalKey::Code(code) = key.physical_key {
            if key.state == winit::event::ElementState::Released {
                self.is_pressing.insert(code, false);
            } else {
                self.is_pressing.insert(code, true);
            }
        }
    }

    pub fn is_pressed(&self, key: &KeyCode) -> bool {
        self.is_pressing.get(key).copied().unwrap_or(false)
    }
}
