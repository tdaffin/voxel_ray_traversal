use nalgebra::Vector3;
use winit::{
    event::{DeviceEvent, MouseButton, WindowEvent},
    keyboard::KeyCode,
    window::{CursorGrabMode, Window},
};
use winit_input_helper::WinitInputHelper;

use crate::camera::Camera;

pub struct InputController {
    helper: WinitInputHelper,
    pub focused: bool,
}

impl InputController {
    pub fn new(helper: WinitInputHelper) -> Self {
        Self { helper, focused: false }
    }

    pub fn helper(&self) -> &WinitInputHelper {
        &self.helper
    }
    pub fn helper_mut(&mut self) -> &mut WinitInputHelper {
        &mut self.helper
    }

    /// Process per-frame update: returns whether an exit was requested.
    pub fn update(&mut self, camera: &mut Camera, window: &Window) {
        if self.focused {
            if let Some(dt) =
                self.helper.delta_time().as_ref().map(std::time::Duration::as_secs_f64)
            {
                let t = |k: KeyCode| self.helper.key_held(k) as u8 as f64;
                let v = Vector3::new(KeyCode::KeyD, KeyCode::KeyW, KeyCode::KeyQ).map(t)
                    - Vector3::new(KeyCode::KeyA, KeyCode::KeyS, KeyCode::KeyE).map(t);
                camera.position += (camera.rotation_matrix() * v.push(0.0) * dt).xyz();
                let sens = 0.001 * (camera.fov.to_radians() * 0.5).tan();
                let (dx, dy) = self.helper.mouse_diff();
                camera.rotation.z -= dx as f64 * sens;
                camera.rotation.x -= dy as f64 * sens;
                camera.rotation.x = camera
                    .rotation
                    .x
                    .clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
                camera.rotation.y = camera.rotation.y.rem_euclid(std::f64::consts::TAU);
                let ds = self.helper.scroll_diff();
                let tanfov = (camera.fov.to_radians() * 0.5).tan();
                camera.fov = ((tanfov * (ds.1 as f64 * -0.1).exp()).atan() * 2.0).to_degrees();
            }
        }
        if self.helper.mouse_pressed(MouseButton::Left) && !self.focused {
            self.focused = true;
            window.set_cursor_grab(CursorGrabMode::Confined).ok();
            window.set_cursor_visible(false);
        }
        if self.helper.key_pressed(KeyCode::Escape) && self.focused {
            self.focused = false;
            window.set_cursor_grab(CursorGrabMode::None).ok();
            window.set_cursor_visible(true);
        }
    }

    pub fn process_window_event(&mut self, event: &WindowEvent) {
        self.helper.process_window_event(event);
    }
    pub fn process_device_event(&mut self, event: &DeviceEvent) {
        self.helper.process_device_event(event);
    }
    pub fn step(&mut self) {
        self.helper.step();
    }
    pub fn end_step(&mut self) {
        self.helper.end_step();
    }
    pub fn close_requested(&self) -> bool {
        self.helper.close_requested()
    }
    pub fn window_resized(&self) -> Option<(u32, u32)> {
        self.helper.window_resized().map(|ps| (ps.width, ps.height))
    }
}
