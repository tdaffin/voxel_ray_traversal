use winit::event_loop::EventLoop;

mod app;
mod benchmark;
mod camera;
mod frame_renderer;
mod frame_timer;
mod gpu;
mod hot_reload;
mod input_controller;
mod model;
mod pipelines;
mod push_constants;
mod render_mode;
mod rendering;
mod swapchain_flow;
mod swapchain_resources;
mod ui;
mod voxel;
mod voxel_job;
mod voxelize;
use app::App;

// RenderContext moved to rendering module

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app).unwrap();
}
