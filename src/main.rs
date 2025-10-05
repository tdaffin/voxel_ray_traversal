use winit::event_loop::EventLoop;

mod app;
mod benchmark;
mod camera;
mod frame_timer;
mod hot_reload;
mod model;
mod push_constants;
mod render_mode;
mod rendering;
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
