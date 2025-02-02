use super::{
    keyboard::Keyboard,
    mouse::Mouse,
};

#[derive(Debug, Default)]
pub struct Control {
    keyboard: Keyboard,
    mouse: Mouse,
}

impl Control {
    pub fn keyboard(&self) -> &Keyboard {
        &self.keyboard
    }

    pub fn keyboard_mut(&mut self) -> &mut Keyboard {
        &mut self.keyboard
    }

    pub fn mouse(&self) -> &Mouse {
        &self.mouse
    }

    pub fn mouse_mut(&mut self) -> &mut Mouse {
        &mut self.mouse
    }
}
