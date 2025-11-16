use nalgebra::Vector3;
use std::sync::Arc;
use winit::{
    event::{DeviceEvent, MouseButton, WindowEvent},
    keyboard::KeyCode,
    window::{CursorGrabMode, Window},
};
use winit_input_helper::WinitInputHelper;

use crate::{camera::Camera, zoom_depth_sampler::ZoomDepthSampler};
use vulkano::image::Image;

pub struct InputController {
    helper: WinitInputHelper,
    pub focused: bool,
}

pub struct ZoomSampleContext<'a> {
    pub sampler: &'a mut ZoomDepthSampler,
    pub depth_image: Arc<Image>,
    pub render_extent: [u32; 2],
}

impl InputController {
    pub fn new(helper: WinitInputHelper) -> Self {
        Self { helper, focused: false }
    }

    pub fn helper(&self) -> &WinitInputHelper {
        &self.helper
    }

    /// Process per-frame update: returns whether an exit was requested.
    pub fn update(
        &mut self, camera: &mut Camera, window: &Window,
        zoom_context: Option<&mut ZoomSampleContext<'_>>,
    ) {
        if self.focused {
            if let Some(dt) =
                self.helper.delta_time().as_ref().map(std::time::Duration::as_secs_f64)
            {
                let t = |k: KeyCode| self.helper.key_held(k) as u8 as f64;
                let v = Vector3::new(KeyCode::KeyD, KeyCode::KeyW, KeyCode::KeyQ).map(t)
                    - Vector3::new(KeyCode::KeyA, KeyCode::KeyS, KeyCode::KeyE).map(t);
                camera.position += (camera.rotation_matrix() * v.push(0.0) * dt).xyz();
                let sens = camera.mouse_sensitivity * (camera.fov.to_radians() * 0.5).tan();
                let (dx, dy) = self.helper.mouse_diff();
                camera.rotation.z -= dx as f64 * sens;
                camera.rotation.x -= dy as f64 * sens;
                camera.rotation.x = camera
                    .rotation
                    .x
                    .clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
                camera.rotation.y = camera.rotation.y.rem_euclid(std::f64::consts::TAU);
                let ds = self.helper.scroll_diff();
                //let tanfov = (camera.fov.to_radians() * 0.5).tan();
                //camera.fov = ((tanfov * (ds.1 as f64 * -0.1).exp()).atan() * 2.0).to_degrees();
                if ds.1 != 0.0 {
                    let scroll_delta = ds.1 as f64;
                    // Dolly by moving along the current forward direction instead of changing FOV.
                    let forward =
                        (camera.rotation_matrix() * Vector3::new(0.0, 1.0, 0.0).push(0.0)).xyz();
                    let zoom_speed = 0.2;
                    let mut move_amount = scroll_delta * zoom_speed;
                    if move_amount > 0.0 {
                        if let Some(ctx) = zoom_context {
                            if ctx.render_extent[0] > 0 && ctx.render_extent[1] > 0 {
                                let focus = [ctx.render_extent[0] / 2, ctx.render_extent[1] / 2];
                                if let Some(depth) = ctx.sampler.sample(
                                    ctx.depth_image.clone(),
                                    ctx.render_extent,
                                    focus,
                                ) {
                                    let safety_margin = 1.0;
                                    let max_step = (depth - safety_margin).max(0.0);
                                    move_amount = move_amount.min(max_step);
                                    //move_amount = depth/10.0;
                                }
                            }
                        }
                    }
                    camera.position += forward * move_amount;
                }
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
    pub fn window_resized(&self) -> Option<(u32, u32)> {
        self.helper.window_resized().map(|ps| (ps.width, ps.height))
    }
}
