use nalgebra::Vector3;
use std::path::PathBuf;
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoop;

use crate::{
    app::App, camera::Camera, frame_timer::FrameTimer, gpu::GpuContext,
    input_controller::InputController, model::Model, pipelines::PipelineManager,
    render_mode::RenderMode, voxel_facade::VoxelSystem,
};
use winit_input_helper::WinitInputHelper;

const DEFAULT_INITIAL_VOXEL_RESOLUTION: u32 = 24;
const DEFAULT_WINDOW_RESOLUTION: PhysicalSize<u32> = PhysicalSize::new(960, 960);

/// Builder for `App` allowing customization of initial parameters without
/// growing `App::new`.
#[allow(dead_code)] // Builder setters may be unused in some binaries until customization is added
pub struct AppBuilder {
    pub initial_voxel_resolution: u32,
    pub model: Model,
    pub render_mode: RenderMode,
    pub render_scale: f32,
    pub window_resolution: PhysicalSize<u32>,
    pub camera_fov: f64,
}

impl Default for AppBuilder {
    fn default() -> Self {
        Self {
            initial_voxel_resolution: DEFAULT_INITIAL_VOXEL_RESOLUTION,
            model: Model::Bunny,
            render_mode: RenderMode::Shade,
            render_scale: 1.0,
            window_resolution: DEFAULT_WINDOW_RESOLUTION,
            camera_fov: 35.0,
        }
    }
}

#[allow(dead_code)]
impl AppBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn voxel_resolution(mut self, r: u32) -> Self {
        self.initial_voxel_resolution = r;
        self
    }
    pub fn model(mut self, m: Model) -> Self {
        self.model = m;
        self
    }
    pub fn render_mode(mut self, rm: RenderMode) -> Self {
        self.render_mode = rm;
        self
    }
    pub fn render_scale(mut self, s: f32) -> Self {
        self.render_scale = s;
        self
    }
    pub fn window_resolution(mut self, w: PhysicalSize<u32>) -> Self {
        self.window_resolution = w;
        self
    }
    pub fn camera_fov(mut self, fov: f64) -> Self {
        self.camera_fov = fov;
        self
    }

    pub fn build(self, event_loop: &EventLoop<()>) -> App {
        // GPU + pipelines
        let gpu = GpuContext::new(event_loop);
        let shaders_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
        let pipelines = PipelineManager::new(gpu.device.clone(), &shaders_dir);

        // Voxel system
        let voxel = VoxelSystem::new(
            self.initial_voxel_resolution,
            gpu.memory_allocator.clone(),
            gpu.descriptor_set_allocator.clone(),
            gpu.command_buffer_allocator.clone(),
            gpu.queue.clone(),
            &pipelines.render,
        );
        let future_grid_resolutions = voxel.manager.grid_resolutions.clone();

        // Camera setup: derive layout from number of active grids (initially all models voxelize eventually)
        let grid_count = Model::ALL.len() as f64; // approximate upper bound; actual active may be fewer early
        let base_res = 1.0f64; // normalized unit size per grid before scaling by its own resolution in shader math
        let spacing_scale = 0.25; // must match shader spacing ratio
        let spacing = base_res * spacing_scale;
        let stride = base_res + spacing;
        let total_width = (grid_count - 1.0) * (base_res + spacing) + base_res;
        // Focus roughly on second grid (gives a pleasing angle when more than one present)
        let focus_index = (grid_count.min(2.0) - 1.0).max(0.0);
        let target_x = focus_index * stride + base_res * 0.5;
        let target = Vector3::new(target_x, base_res * 0.5, base_res * 0.5);
        let dist = total_width * 1.1; // scale distance by total span
        let cam_pos = target + Vector3::new(-dist, dist * 0.6, dist * 0.8);
        let mut camera =
            Camera::new(cam_pos, Vector3::zeros(), self.window_resolution.into(), self.camera_fov);
        camera.look_at(target);

        let input = InputController::new(WinitInputHelper::new());

        App::from_parts(
            gpu,
            pipelines,
            voxel,
            future_grid_resolutions,
            self.model,
            camera,
            self.render_mode,
            self.render_scale,
            input,
            FrameTimer::new(),
        )
    }
}
