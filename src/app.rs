use crate::input_controller::InputController;
use egui_winit_vulkano::{Gui, GuiConfig};
use std::{path::PathBuf, sync::Arc};
use vulkano::{
    image::view::{ImageView, ImageViewCreateInfo},
    swapchain::Surface,
    sync::GpuFuture,
};
use winit::dpi::PhysicalSize;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use crate::camera::Camera;
use crate::frame_timer::FrameTimer;
use crate::gpu::GpuContext;
use crate::model_discovery::DiscoveredModel;
use crate::pipelines::PipelineManager;
// push_constants now handled inside frame_renderer
use crate::app_builder::AppBuilder;
use crate::frame_renderer::record_frame;
use crate::render_mode::RenderMode;
use crate::rendering::{RenderContext, get_images_and_sets, get_swapchain_images, load_icon};
use crate::swapchain_manager::SwapchainManager;
use crate::ui_state::{PersistedUiState, default_ui_state_path, load_ui_state, save_ui_state};
use crate::voxel_facade::VoxelSystem;

const INITIAL_WINDOW_RESOLUTION: PhysicalSize<u32> = PhysicalSize::new(960, 960);

pub struct App {
    pub(crate) gpu: GpuContext,

    pub(crate) pipelines: PipelineManager,
    // Voxel subsystem
    pub(crate) voxel: VoxelSystem,
    pub(crate) future_grid_resolutions: Vec<u32>,
    pub(crate) available_models: Vec<DiscoveredModel>,
    pub(crate) add_models_popup_open: bool,
    pub(crate) add_models_selection: Vec<bool>,

    pub(crate) camera: Camera,
    pub(crate) render_mode: RenderMode,
    pub(crate) render_scale: f32,
    pub(crate) light_dir: [f32; 3],
    pub(crate) light_theta: f32,
    pub(crate) light_phi: f32,
    pub(crate) always_instant: bool,
    pub(crate) hit_back: bool,
    pub(crate) verbose_logging: bool,
    pub(crate) advanced_window_open: bool,

    input: InputController,
    frame_timer: FrameTimer,

    ui_state_dirty: bool,
    ui_state_path: Option<PathBuf>,

    pub(crate) rcx: Option<RenderContext>,
}

impl App {
    pub(crate) fn from_parts(
        gpu: GpuContext, pipelines: PipelineManager, voxel: VoxelSystem,
        future_grid_resolutions: Vec<u32>, available_models: Vec<DiscoveredModel>, camera: Camera,
        render_mode: RenderMode, render_scale: f32, input: InputController,
        frame_timer: FrameTimer, verbose_logging: bool,
    ) -> Self {
        let ui_state_path = default_ui_state_path();
        let add_models_selection = vec![false; available_models.len()];
        let mut app = App {
            gpu,
            pipelines,
            voxel,
            future_grid_resolutions,
            available_models,
            add_models_popup_open: false,
            add_models_selection,
            camera,
            render_mode,
            render_scale,
            light_dir: [0.5, 0.8, 0.3],
            light_theta: 0.9,
            light_phi: 0.6,
            always_instant: false,
            hit_back: false,
            verbose_logging,
            advanced_window_open: false,
            input,
            frame_timer,
            ui_state_dirty: false,
            ui_state_path: ui_state_path.clone(),
            rcx: None,
        };
        app.update_light_direction_from_angles();
        if let Some(path) = ui_state_path {
            if let Some(saved) = load_ui_state(&path) {
                app.apply_persisted_state(&saved);
            }
        }
        crate::log_config::set_verbose_logging(app.verbose_logging);
        app
    }
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        AppBuilder::default().build(event_loop)
    }

    pub fn fps(&self) -> u32 {
        self.frame_timer.fps()
    }

    pub(crate) fn model_key(model: &DiscoveredModel) -> String {
        format!("{}.{}", model.name, model.extension)
    }

    pub(crate) fn ensure_add_model_selection_len(&mut self) {
        if self.add_models_selection.len() != self.available_models.len() {
            self.add_models_selection = vec![false; self.available_models.len()];
        }
    }

    fn models_from_keys(&self, keys: &[String]) -> Vec<DiscoveredModel> {
        let mut out = Vec::new();
        for key in keys {
            if let Some(model) = self.available_models.iter().find(|m| Self::model_key(m) == *key) {
                out.push(model.clone());
            } else {
                eprintln!("[ui] persisted model '{key}' no longer available; skipping");
            }
        }
        out
    }

    pub(crate) fn set_loaded_models(&mut self, models: Vec<DiscoveredModel>) {
        let placeholder_resolution = self.voxel.manager.voxel_resolution.max(8);
        self.voxel.set_models(models, &self.pipelines.render, placeholder_resolution);
        self.future_grid_resolutions = self.voxel.manager.grid_resolutions.clone();
        self.voxel.manager.future_grid_resolutions = self.future_grid_resolutions.clone();
        let max_grids = (self.voxel.manager.models.len() + 1).max(1) as u32;
        self.voxel.manager.active_voxel_grids =
            self.voxel.manager.active_voxel_grids.clamp(1, max_grids);
        self.ensure_add_model_selection_len();
    }

    pub(crate) fn add_models_to_scene(&mut self, models: Vec<DiscoveredModel>) {
        if models.is_empty() {
            return;
        }
        let placeholder_resolution = self.voxel.manager.voxel_resolution.max(8);
        self.voxel.add_models(models, &self.pipelines.render, placeholder_resolution);
        self.future_grid_resolutions = self.voxel.manager.grid_resolutions.clone();
        self.voxel.manager.future_grid_resolutions = self.future_grid_resolutions.clone();
        let max_grids = (self.voxel.manager.models.len() + 1).max(1) as u32;
        self.voxel.manager.active_voxel_grids =
            self.voxel.manager.active_voxel_grids.clamp(1, max_grids);
        self.ensure_add_model_selection_len();
    }

    // start_background_voxelization now handled by VoxelManager

    fn update(&mut self, event_loop: &ActiveEventLoop) {
        // Advance frame timer (store happens internally, UI reads via frame_timer.fps())
        self.frame_timer.frame();
        if self.input.helper().close_requested() {
            event_loop.exit();
            return;
        }
        let rcx = self.rcx.as_mut().unwrap();
        // Update input (camera movement, focus toggles)
        self.input.update(&mut self.camera, &rcx.window);
        // Drain any completed voxelization results and upload to GPU
        let delta_seconds =
            self.input.helper().delta_time().map(|d| d.as_secs_f32()).unwrap_or(0.0);
        self.voxel.poll(&self.pipelines.render, delta_seconds);
    }

    fn render(&mut self, _event_loop: &ActiveEventLoop) {
        self.pipelines.maybe_reload();

        {
            let rcx = self.rcx.as_mut().unwrap();
            if self.input.window_resized().is_some() {
                rcx.recreate_swapchain = true;
            }
            if rcx.window.inner_size().width == 0 || rcx.window.inner_size().height == 0 {
                return;
            }
            SwapchainManager::ensure_resources(rcx, &self.gpu, &self.pipelines, self.render_scale);
        }

        let (image_index, acquire_future) = {
            let rcx = self.rcx.as_mut().unwrap();
            match SwapchainManager::acquire(rcx) {
                Some(r) => {
                    if r.suboptimal {
                        rcx.recreate_swapchain = true;
                    }
                    (r.image_index, r.future)
                }
                None => return,
            }
        };

        let (request_regen_voxels, trigger_benchmark) = self.draw_ui();

        if request_regen_voxels {
            // Copy future per-grid resolutions into active ones (truncate/extend safely)
            self.voxel.manager.future_grid_resolutions = self.future_grid_resolutions.clone();
            self.voxel.regenerate(&self.pipelines.render, 8);
        }

        if trigger_benchmark {
            crate::benchmark_hook::run_bench_if_requested(
                true,
                30,
                &self.camera,
                &self.voxel.manager,
                self.render_mode as u32,
                self.always_instant,
                self.hit_back,
                &self.pipelines,
                self.gpu.queue.clone(),
                self.gpu.command_buffer_allocator.clone(),
                self.rcx.as_mut().unwrap(),
                self.gpu.device.clone(),
            );
        }

        // Build command buffer via renderer helper
        let rcx = self.rcx.as_mut().unwrap();
        let outputs = record_frame(
            &self.gpu,
            &self.pipelines,
            rcx,
            &self.voxel.manager,
            &mut self.camera,
            self.render_mode as u32,
            image_index,
            self.light_dir,
            self.always_instant,
            self.hit_back,
        );

        let render_future =
            acquire_future.then_execute(self.gpu.queue.clone(), outputs.command_buffer).unwrap();

        let gui_future =
            rcx.gui.draw_on_image(render_future, rcx.image_views[image_index as usize].clone());

        SwapchainManager::present_and_wait(&self.gpu, rcx, image_index, gui_future);
    }

    pub(crate) fn update_light_direction_from_angles(&mut self) {
        let theta = self.light_theta;
        let phi = self.light_phi;
        let ct = theta.cos();
        self.light_dir = [ct * phi.cos(), theta.sin(), ct * phi.sin()];
    }

    fn apply_persisted_state(&mut self, state: &PersistedUiState) {
        let desired_models = self.models_from_keys(&state.loaded_models);
        self.set_loaded_models(desired_models);

        self.render_mode = state.render_mode;
        self.always_instant = state.always_instant;
        self.hit_back = state.hit_back;
        self.verbose_logging = state.verbose_logging;
        self.render_scale = state.render_scale.clamp(0.125, 8.0);
        self.camera.fov = state.camera_fov.clamp(0.0, 180.0);
        self.camera.mouse_sensitivity = state.mouse_sensitivity.clamp(0.0005_f64, 0.01_f64);
        self.light_theta = state.light_theta.clamp(-1.57, 1.57);
        self.light_phi = state.light_phi.clamp(-std::f32::consts::PI, std::f32::consts::PI);
        self.advanced_window_open = state.advanced_window_open;

        let max_grids = (self.voxel.manager.models.len() + 1).max(1) as u32;
        self.voxel.manager.active_voxel_grids = state.active_voxel_grids.clamp(1, max_grids);

        self.future_grid_resolutions = self.voxel.manager.grid_resolutions.clone();
        if !state.future_grid_resolutions.is_empty() {
            for (dst, src) in
                self.future_grid_resolutions.iter_mut().zip(state.future_grid_resolutions.iter())
            {
                let snapped = ((*src).max(8) + 7) / 8 * 8;
                *dst = snapped;
            }
        }
        self.voxel.manager.future_grid_resolutions = self.future_grid_resolutions.clone();

        self.update_light_direction_from_angles();
        self.ui_state_dirty = false;
    }

    fn current_persisted_state(&self) -> PersistedUiState {
        PersistedUiState {
            render_mode: self.render_mode,
            always_instant: self.always_instant,
            hit_back: self.hit_back,
            verbose_logging: self.verbose_logging,
            render_scale: self.render_scale,
            active_voxel_grids: self.voxel.manager.active_voxel_grids,
            camera_fov: self.camera.fov,
            mouse_sensitivity: self.camera.mouse_sensitivity,
            light_theta: self.light_theta,
            light_phi: self.light_phi,
            advanced_window_open: self.advanced_window_open,
            future_grid_resolutions: self.future_grid_resolutions.clone(),
            loaded_models: self.voxel.manager.models.iter().map(App::model_key).collect(),
        }
    }

    pub(crate) fn mark_ui_state_dirty(&mut self) {
        self.ui_state_dirty = true;
    }

    pub(crate) fn flush_ui_state(&mut self) {
        if !self.ui_state_dirty {
            return;
        }
        if let Some(path) = self.ui_state_path.clone() {
            let state = self.current_persisted_state();
            if let Err(err) = save_ui_state(&path, &state) {
                eprintln!("[ui] failed to save ui state: {err}");
            } else {
                self.ui_state_dirty = false;
            }
        }
    }

    pub(crate) fn apply_render_scale_change(&mut self) {
        if let Some(rcx) = self.rcx.as_mut() {
            let window_extent: [u32; 2] = rcx.window.inner_size().into();
            let render_extent = [
                (window_extent[0] as f32 * self.render_scale) as u32,
                (window_extent[1] as f32 * self.render_scale) as u32,
            ];
            (rcx.render_image, rcx.render_set, rcx.resample_image, rcx.resample_set) =
                get_images_and_sets(
                    self.gpu.memory_allocator.clone(),
                    self.gpu.descriptor_set_allocator.clone(),
                    &self.pipelines.render,
                    &self.pipelines.resample,
                    render_extent,
                    window_extent,
                );
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(INITIAL_WINDOW_RESOLUTION)
                        .with_window_icon(Some(load_icon(include_bytes!("../assets/icon.png"))))
                        .with_title("Voxel Ray Traversal"),
                )
                .unwrap(),
        );
        let surface = Surface::from_window(self.gpu.instance.clone(), window.clone()).unwrap();

        let (swapchain, images) = get_swapchain_images(&self.gpu.device, &surface, &window);
        let image_views = images
            .iter()
            .map(|i| ImageView::new(i.clone(), ImageViewCreateInfo::from_image(i)).unwrap())
            .collect::<Vec<_>>();

        let window_extent: [u32; 2] = window.inner_size().into();
        let render_extent = [
            (window_extent[0] as f32 * self.render_scale) as u32,
            (window_extent[1] as f32 * self.render_scale) as u32,
        ];
        let (render_image, render_set, resample_image, resample_set) = get_images_and_sets(
            self.gpu.memory_allocator.clone(),
            self.gpu.descriptor_set_allocator.clone(),
            &self.pipelines.render,
            &self.pipelines.resample,
            render_extent,
            window_extent,
        );

        let gui = Gui::new(
            event_loop,
            surface,
            self.gpu.queue.clone(),
            swapchain.image_format(),
            GuiConfig { is_overlay: true, ..Default::default() },
        );

        let recreate_swapchain = false;

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            image_views,
            render_image,
            render_set,
            resample_image,
            resample_set,
            gui,
            recreate_swapchain,
        });
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        self.input.step();
    }

    fn window_event(
        &mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent,
    ) {
        if !self.rcx.as_mut().unwrap().gui.update(&event) {
            self.input.process_window_event(&event);
        }

        if event == WindowEvent::RedrawRequested {
            self.render(event_loop);
        }
    }

    fn device_event(
        &mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent,
    ) {
        self.input.process_device_event(&event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.input.end_step();
        self.update(event_loop);
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}
