use crate::input_controller::InputController;
use egui_winit_vulkano::{Gui, GuiConfig};
use std::sync::Arc;
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
use crate::model::Model;
use crate::pipelines::PipelineManager;
// push_constants now handled inside frame_renderer
use crate::app_builder::AppBuilder;
use crate::frame_renderer::record_frame;
use crate::render_mode::RenderMode;
use crate::rendering::{RenderContext, get_images_and_sets, get_swapchain_images, load_icon};
use crate::swapchain_manager::SwapchainManager;
use crate::voxel_facade::VoxelSystem;

const INITIAL_WINDOW_RESOLUTION: PhysicalSize<u32> = PhysicalSize::new(960, 960);

pub struct App {
    pub(crate) gpu: GpuContext,

    pub(crate) pipelines: PipelineManager,
    // Voxel subsystem
    pub(crate) voxel: VoxelSystem,
    pub(crate) future_grid_resolutions: Vec<u32>,
    pub(crate) model: Model,

    pub(crate) camera: Camera,
    pub(crate) render_mode: RenderMode,
    pub(crate) render_scale: f32,

    input: InputController,
    frame_timer: FrameTimer,
    pub(crate) fps: u32, // kept for external access; mirrors frame_timer.fps()

    pub(crate) rcx: Option<RenderContext>,
}

impl App {
    pub(crate) fn from_parts(
        gpu: GpuContext, pipelines: PipelineManager, voxel: VoxelSystem,
        future_grid_resolutions: Vec<u32>, model: Model, camera: Camera, render_mode: RenderMode,
        render_scale: f32, input: InputController, frame_timer: FrameTimer,
    ) -> Self {
        App { gpu, pipelines, voxel, future_grid_resolutions, model, camera, render_mode,
            render_scale, input, frame_timer, fps: 0, rcx: None,
        }
    }
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        AppBuilder::default().build(event_loop)
    }

    // start_background_voxelization now handled by VoxelManager

    fn update(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(fps) = self.frame_timer.frame() {
            self.fps = fps;
        }
        if self.input.helper().close_requested() {
            event_loop.exit();
            return;
        }
        let rcx = self.rcx.as_mut().unwrap();
        // Update input (camera movement, focus toggles)
        self.input.update(&mut self.camera, &rcx.window);
        // Drain any completed voxelization results and upload to GPU
        self.voxel.poll(&self.pipelines.render);
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
            crate::benchmark_hook::run_bench_if_requested(true, 30, &self.camera,
                &self.voxel.manager, self.render_mode as u32, &self.pipelines,
                self.gpu.queue.clone(), self.gpu.command_buffer_allocator.clone(),
                self.rcx.as_mut().unwrap(), self.gpu.device.clone(),
            );
        }

        // Build command buffer via renderer helper
        let rcx = self.rcx.as_mut().unwrap();
        let outputs = record_frame(&self.gpu, &self.pipelines, rcx,
            &self.voxel.manager, &mut self.camera, self.render_mode as u32, image_index,
        );

        let render_future =
            acquire_future.then_execute(self.gpu.queue.clone(), outputs.command_buffer).unwrap();

        let gui_future =
            rcx.gui.draw_on_image(render_future, rcx.image_views[image_index as usize].clone());

        SwapchainManager::present_and_wait(&self.gpu, rcx, image_index, gui_future);
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

        let gui = Gui::new(event_loop, surface, self.gpu.queue.clone(),
            swapchain.image_format(), 
            GuiConfig { is_overlay: true, ..Default::default() },
        );

        let recreate_swapchain = false;

        self.rcx = Some(RenderContext { window, swapchain, image_views, render_image, render_set,
            resample_image, resample_set, gui, recreate_swapchain,
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
