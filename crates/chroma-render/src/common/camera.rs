use crate::control::{
    keyboard::Keyboard,
    mouse::Mouse,
    Control,
};
use nalgebra_glm::{
    Mat4,
    Vec1,
    Vec2,
    Vec3,
    Vec4,
};
use std::time::Duration;
use winit::{
    dpi::PhysicalPosition,
    event::MouseButton,
    keyboard::KeyCode,
};

const LOOK_AT: Vec4 = Vec4::new(0.0, 0.0, 1.0, 0.0);
const CAM_RIGHT: Vec4 = Vec4::new(1.0, 0.0, 0.0, 0.0);
const CAM_UP: Vec4 = Vec4::new(0.0, 1.0, 0.0, 0.0);

fn apply_rotation(rotation: &Vec2, look_at: &mut Vec4, cam_right: &mut Vec4, cam_up: &mut Vec4) {
    let pitch_mat = nalgebra_glm::rotate(&Mat4::identity(), rotation.y, &Vec3::new(1.0, 0.0, 0.0));
    let yaw_mat = nalgebra_glm::rotate(&Mat4::identity(), rotation.x, &Vec3::new(0.0, 1.0, 0.0));
    let rotation_mat = yaw_mat * pitch_mat;

    *look_at = nalgebra_glm::normalize(&(rotation_mat * LOOK_AT));
    *cam_right = rotation_mat * CAM_RIGHT;

    let look_at_vec3 = look_at.xyz();
    let cam_right_vec3 = cam_right.xyz();
    let cam_up_vec3 = nalgebra_glm::cross(&look_at_vec3, &cam_right_vec3);
    *cam_up = Vec4::new(cam_up_vec3.x, cam_up_vec3.y, cam_up_vec3.z, 0.0);
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
#[allow(dead_code)]
pub struct TransformParams {
    pub world: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
    pub view_proj: Mat4,
    pub world_view_proj: Mat4,
    pub proj_to_world: Mat4,
    pub view_inv: Mat4,
    pub proj_inv: Mat4,
}

#[derive(Clone, Debug, Default)]
#[repr(C)]
#[allow(dead_code)]
pub struct CameraMatrixParams {
    view_inv: Mat4,
    proj_inv: Mat4,
}

#[derive(Clone, Debug, Default)]
#[repr(C)]
#[allow(dead_code)]
pub struct CameraParams {
    pub position: Vec3,
}

#[derive(Clone, Debug)]
pub struct Camera {
    position: Vec3,
    rotation: Vec2,
    look_at: Vec4,
    cam_right: Vec4,
    cam_up: Vec4,
    world: Mat4,
    proj: Mat4,
    mouse_sens: f32,
    move_speed: f32,
}

impl Camera {
    pub fn new(width: u32, height: u32) -> Self {
        const FOV: f32 = 60.0;
        const NEAR_Z: f32 = 0.1;
        const FAR_Z: f32 = 1000.0;
        let fov_rad = nalgebra_glm::radians(&Vec1::new(FOV)).x;

        let mut proj = nalgebra_glm::perspective_fov_rh_zo(
            fov_rad,
            width as f32,
            height as f32,
            NEAR_Z,
            FAR_Z,
        );
        proj[(1, 1)] *= -1.0;

        let position = Vec3::new(1.0, 0.82, 12.0);
        let rotation = Vec2::new(3.14, 0.0);

        let mut look_at = LOOK_AT;
        let mut cam_right = CAM_RIGHT;
        let mut cam_up = CAM_UP;
        apply_rotation(&rotation, &mut look_at, &mut cam_right, &mut cam_up);

        Self {
            position,
            rotation,
            mouse_sens: 0.1,
            move_speed: 0.3,
            look_at,
            cam_right,
            cam_up,
            world: Mat4::identity(),
            proj,
        }
    }

    fn input_mouse(
        &mut self,
        mouse: &Mouse,
        window_center: PhysicalPosition<f32>,
        elapsed: Duration,
    ) {
        if !mouse.is_toggled(&MouseButton::Right) {
            return;
        }
        let mouse_position = mouse.position_from_center(window_center);

        const SENS_PER_SEC: f32 = 100.0;
        let sens_ratio = elapsed.as_secs_f32() * SENS_PER_SEC;
        let x_offset = mouse_position.x * self.mouse_sens * sens_ratio;
        let y_offset = mouse_position.y * self.mouse_sens * sens_ratio;

        self.rotation.x -= x_offset;
        self.rotation.y += y_offset;

        const EPSILON: f32 = 1e-6;
        self.rotation.y = self.rotation.y.clamp(
            -std::f32::consts::FRAC_PI_2 + EPSILON,
            std::f32::consts::FRAC_PI_2 - EPSILON,
        );

        let rotation = self.rotation;
        apply_rotation(
            &rotation,
            &mut self.look_at,
            &mut self.cam_right,
            &mut self.cam_up,
        );
    }

    fn input_keyboard(&mut self, keyboard: &Keyboard, elapsed: Duration) {
        let pressed_w = keyboard.is_pressed(&KeyCode::KeyW);
        let pressed_s = keyboard.is_pressed(&KeyCode::KeyS);
        let pressed_a = keyboard.is_pressed(&KeyCode::KeyA);
        let pressed_d = keyboard.is_pressed(&KeyCode::KeyD);
        let pressed_q = keyboard.is_pressed(&KeyCode::KeyQ);
        let pressed_e = keyboard.is_pressed(&KeyCode::KeyE);
        let pressed_alt_left = keyboard.is_pressed(&KeyCode::AltLeft);

        let move_forward = pressed_w as i32 - pressed_s as i32;
        let move_right = pressed_d as i32 - pressed_a as i32;
        let move_up = pressed_e as i32 - pressed_q as i32;

        const SPEED_PER_SEC: f32 = 0.1;
        let speed_ratio = {
            let mut ratio = elapsed.as_micros() as f32 * SPEED_PER_SEC;
            if pressed_alt_left {
                ratio *= 0.25;
            }
            ratio
        };
        self.position += self.look_at.xyz() * move_forward as f32 * self.move_speed * speed_ratio;
        self.position -= self.cam_right.xyz() * move_right as f32 * self.move_speed * speed_ratio;
        self.position += self.cam_up.xyz() * move_up as f32 * self.move_speed * speed_ratio;
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn input_control(
        &mut self,
        control: &Control,
        window_center: PhysicalPosition<f32>,
        elapsed: Duration,
    ) {
        self.input_mouse(control.mouse(), window_center, elapsed);
        self.input_keyboard(control.keyboard(), elapsed);
    }

    pub fn set_move_speed(&mut self, move_speed: f32) {
        self.move_speed = move_speed;
    }

    pub fn move_speed(&mut self) -> f32 {
        self.move_speed
    }

    pub fn set_mouse_sens(&mut self, mouse_sens: f32) {
        self.mouse_sens = mouse_sens;
    }

    pub fn mouse_sens(&mut self) -> f32 {
        self.mouse_sens
    }

    pub fn create_view_matrix(&self) -> Mat4 {
        let look_at = self.look_at.xyz();
        let cam_up = self.cam_up.xyz();
        nalgebra_glm::look_at(&self.position, &(self.position + look_at), &cam_up)
    }

    pub fn create_transform_params(&self) -> TransformParams {
        let view_matrix = self.create_view_matrix();
        let proj_to_world_matrix = nalgebra_glm::inverse(&(self.proj * view_matrix));

        TransformParams {
            world: self.world,
            view: view_matrix,
            proj: self.proj,
            view_proj: self.proj * view_matrix,
            world_view_proj: self.proj * view_matrix * self.world,
            proj_to_world: proj_to_world_matrix,
            view_inv: nalgebra_glm::inverse(&view_matrix),
            proj_inv: nalgebra_glm::inverse(&self.proj),
        }
    }

    pub fn create_camera_params(&self) -> CameraParams {
        CameraParams {
            position: self.position,
        }
    }
}
