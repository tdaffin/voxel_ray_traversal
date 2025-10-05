use winit::event_loop::EventLoop;

mod camera;
mod hot_reload;
mod model;
mod render_mode;
mod rendering;
mod voxel;
mod voxelize;
mod voxel_job;
mod app;
use app::App;

// RenderContext moved to rendering module

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app).unwrap();
}
