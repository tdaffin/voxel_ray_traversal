use egui_winit_vulkano::{Gui, GuiConfig};
use nalgebra::Vector3;
use std::path::PathBuf;
use std::{
    f64::consts::{FRAC_PI_2, TAU},
    sync::Arc,
    time::{Duration, Instant},
};
use vulkano::{
    Validated, Version, VulkanError, VulkanLibrary,
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, ClearColorImageInfo, CommandBufferUsage,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    image::{
        sampler::Filter,
        view::{ImageView, ImageViewCreateInfo},
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{Pipeline, PipelineBindPoint},
    swapchain::{Surface, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image},
    sync::GpuFuture,
};
use winit::dpi::PhysicalSize;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::KeyCode,
    window::{CursorGrabMode, Window, WindowId},
};
use winit_input_helper::WinitInputHelper;

use crate::camera::Camera;
use crate::hot_reload::HotReloadComputePipeline;
use crate::model::Model;
use crate::push_constants::{PushConstantsInput, build_push_constants};
use crate::render_mode::RenderMode;
use crate::rendering::{
    RenderContext, get_allocators, get_images_and_sets, get_swapchain_images, load_icon,
};
use crate::voxel_job::VoxelManager;

const INITIAL_VOXEL_RESOLUTION: u32 = 24;
const INITIAL_WINDOW_RESOLUTION: PhysicalSize<u32> = PhysicalSize::new(960, 960);

pub struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,

    pub(crate) memory_allocator: Arc<StandardMemoryAllocator>,
    pub(crate) descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    pub(crate) render_pipeline: HotReloadComputePipeline,
    pub(crate) resample_pipeline: HotReloadComputePipeline,
    render_pipeline_branchless: HotReloadComputePipeline,
    // Voxel subsystem
    pub(crate) voxel: VoxelManager,
    pub(crate) future_grid_resolutions: Vec<u32>,
    pub(crate) model: Model,

    pub(crate) camera: Camera,
    pub(crate) render_mode: RenderMode,
    pub(crate) render_scale: f32,

    input: WinitInputHelper,
    focused: bool,
    last_second: Instant,
    frames_since_last_second: u32,
    pub(crate) fps: u32,

    pub(crate) rcx: Option<RenderContext>,
}

impl App {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().unwrap();

        let mut required_extensions = Surface::required_extensions(event_loop).unwrap();

        required_extensions.ext_debug_utils = true;

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let mut device_extensions =
            DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::empty() };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
            })
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    dynamic_rendering: true,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let (memory_allocator, descriptor_set_allocator, command_buffer_allocator) =
            get_allocators(&device);

        let shaders_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
        let render_pipeline =
            HotReloadComputePipeline::new(device.clone(), &shaders_dir.join("traverse.comp"));
        let render_pipeline_branchless = HotReloadComputePipeline::with_defines(
            device.clone(),
            &shaders_dir.join("traverse.comp"),
            vec![("BRANCHLESS_TRAVERSAL".to_string(), None::<String>)],
        );
        let resample_pipeline =
            HotReloadComputePipeline::new(device.clone(), &shaders_dir.join("resample.comp"));

        let voxel = VoxelManager::new(
            INITIAL_VOXEL_RESOLUTION,
            memory_allocator.clone(),
            descriptor_set_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            &render_pipeline,
        );
        let future_grid_resolutions = voxel.grid_resolutions.clone();
        let model = Model::Bunny;

        let input = WinitInputHelper::new();

        // Camera setup to frame all three grids.
        let res_f = 1.0f64;
        let spacing = res_f * 0.25; // matches shader
        let stride = res_f + spacing;
        let target = Vector3::new(stride, res_f * 0.5, res_f * 0.5);
        let total_width = 2.0 * stride + res_f;
        let dist = total_width * 1.2;
        let cam_pos = target + Vector3::new(-dist, dist * 0.6, dist * 0.8);
        let mut camera =
            Camera::new(cam_pos, Vector3::zeros(), INITIAL_WINDOW_RESOLUTION.into(), 35.0);
        camera.look_at(target);

        App {
            instance,
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            render_pipeline,
            render_pipeline_branchless,
            resample_pipeline,
            voxel,
            future_grid_resolutions,
            model,
            camera,
            render_mode: RenderMode::Coord,
            render_scale: 1.0,
            input,
            focused: false,
            last_second: Instant::now(),
            frames_since_last_second: 0,
            fps: 0,
            rcx: None,
        }
    }

    // start_background_voxelization now handled by VoxelManager

    fn update(&mut self, event_loop: &ActiveEventLoop) {
        self.frames_since_last_second += 1;
        let now = Instant::now();
        if now.duration_since(self.last_second) >= Duration::from_secs(1) {
            self.fps = self.frames_since_last_second;
            self.frames_since_last_second = 0;
            self.last_second = now;
        }
        let Some(delta_time) = self.input.delta_time().as_ref().map(Duration::as_secs_f64) else {
            return;
        };
        if self.input.close_requested() {
            event_loop.exit();
            return;
        }
        if self.focused {
            let t = |k: KeyCode| self.input.key_held(k) as u8 as f64;
            let v = Vector3::new(KeyCode::KeyD, KeyCode::KeyW, KeyCode::KeyQ).map(t)
                - Vector3::new(KeyCode::KeyA, KeyCode::KeyS, KeyCode::KeyE).map(t);
            self.camera.position +=
                (self.camera.rotation_matrix() * v.push(0.0) * delta_time).xyz();
            let sens = 0.001 * (self.camera.fov.to_radians() * 0.5).tan();
            let (dx, dy) = self.input.mouse_diff();
            self.camera.rotation.z -= dx as f64 * sens;
            self.camera.rotation.x -= dy as f64 * sens;
            self.camera.rotation.x = self.camera.rotation.x.clamp(-FRAC_PI_2, FRAC_PI_2);
            self.camera.rotation.y = self.camera.rotation.y.rem_euclid(TAU);
            let ds = self.input.scroll_diff();
            let tanfov = (self.camera.fov.to_radians() * 0.5).tan();
            self.camera.fov = ((tanfov * (ds.1 as f64 * -0.1).exp()).atan() * 2.0).to_degrees();
        }
        let rcx = self.rcx.as_mut().unwrap();
        // Drain any completed voxelization results and upload to GPU
        self.voxel.poll(
            self.descriptor_set_allocator.clone(),
            &self.render_pipeline,
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.queue.clone(),
        );
        if self.input.mouse_pressed(MouseButton::Left) {
            self.focused = true;
            rcx.window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
            rcx.window.set_cursor_visible(false);
        }
        if self.input.key_pressed(KeyCode::Escape) {
            self.focused = false;
            rcx.window.set_cursor_grab(CursorGrabMode::None).unwrap();
            rcx.window.set_cursor_visible(true);
        }
    }

    fn render(&mut self, _event_loop: &ActiveEventLoop) {
        self.render_pipeline.maybe_reload();
        self.resample_pipeline.maybe_reload();
        self.render_pipeline_branchless.maybe_reload();

        {
            let rcx = self.rcx.as_mut().unwrap();
            if self.input.window_resized().is_some() {
                rcx.recreate_swapchain = true;
            }
            let window_size = rcx.window.inner_size();

            if window_size.width == 0 || window_size.height == 0 {
                return;
            }

            if rcx.recreate_swapchain {
                let images;
                (rcx.swapchain, images) = rcx
                    .swapchain
                    .recreate(SwapchainCreateInfo {
                        image_extent: window_size.into(),
                        ..rcx.swapchain.create_info()
                    })
                    .unwrap();
                rcx.image_views = images
                    .iter()
                    .map(|i| ImageView::new(i.clone(), ImageViewCreateInfo::from_image(i)).unwrap())
                    .collect();
                let window_extent: [u32; 2] = window_size.into();
                let render_extent = [
                    (window_extent[0] as f32 * self.render_scale) as u32,
                    (window_extent[1] as f32 * self.render_scale) as u32,
                ];
                (rcx.render_image, rcx.render_set, rcx.resample_image, rcx.resample_set) =
                    get_images_and_sets(
                        self.memory_allocator.clone(),
                        self.descriptor_set_allocator.clone(),
                        &self.render_pipeline,
                        &self.resample_pipeline,
                        render_extent,
                        window_extent,
                    );
                rcx.recreate_swapchain = false;
            }
        }

        let (image_index, suboptimal, acquire_future) = {
            let rcx = self.rcx.as_mut().unwrap();
            match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.rcx.as_mut().unwrap().recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            }
        };

        if suboptimal {
            self.rcx.as_mut().unwrap().recreate_swapchain = true;
        }

        let (request_regen_voxels, trigger_benchmark) = self.draw_ui();

        if request_regen_voxels {
            // Copy future per-grid resolutions into active ones (truncate/extend safely)
            self.voxel.future_grid_resolutions = self.future_grid_resolutions.clone();
            self.voxel.regenerate(
                self.descriptor_set_allocator.clone(),
                self.memory_allocator.clone(),
                self.command_buffer_allocator.clone(),
                self.queue.clone(),
                &self.render_pipeline,
                8,
            );
        }

        if trigger_benchmark {
            use crate::benchmark::{BenchmarkContext, run};
            let rcx_ref = self.rcx.as_mut().unwrap();
            let outcome = run(&mut BenchmarkContext {
                frames: 30,
                camera: &self.camera,
                voxel: &self.voxel,
                render_mode: self.render_mode as u32,
                branching_pipeline: self.render_pipeline.clone(),
                branchless_pipeline: self.render_pipeline_branchless.clone(),
                queue: self.queue.clone(),
                command_buffer_allocator: self.command_buffer_allocator.clone(),
                rcx: rcx_ref,
                device: self.device.clone(),
            });
            println!("Benchmark Results ({} frames each):", outcome.frames);
            println!("  Branching traversal avg frame CPU: {:?}", outcome.branching_cpu_avg);
            if let Some(ns) = outcome.branching_gpu_avg_ns {
                println!("  Branching traversal avg frame GPU: {:.3} ms", ns as f64 / 1_000_000.0);
            }
            println!("  Branchless traversal avg frame CPU: {:?}", outcome.branchless_cpu_avg);
            if let Some(ns) = outcome.branchless_gpu_avg_ns {
                println!("  Branchless traversal avg frame GPU: {:.3} ms", ns as f64 / 1_000_000.0);
            }
        }

        // Re-borrow render context for the remainder of the standard render path.
        let rcx = self.rcx.as_mut().unwrap();

        let render_extent = rcx.render_image.extent();
        let resample_extent = rcx.resample_image.extent();
        self.camera.extent = [render_extent[0] as f64, render_extent[1] as f64];

        let push_constants = build_push_constants(PushConstantsInput {
            cam_pixel_to_ray: self.camera.pixel_to_ray_matrix(),
            voxel: &self.voxel,
            render_mode: self.render_mode as u32,
        });

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder.clear_color_image(ClearColorImageInfo::image(rcx.render_image.clone())).unwrap();

        builder
            .bind_pipeline_compute(self.render_pipeline.clone())
            .unwrap()
            .push_constants(self.render_pipeline.layout().clone(), 0, push_constants)
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.render_pipeline.layout().clone(),
                0,
                vec![rcx.render_set.clone(), self.voxel.voxel_set.clone()],
            )
            .unwrap();
        unsafe {
            builder
                .dispatch([render_extent[0].div_ceil(8), render_extent[1].div_ceil(8), 1])
                .unwrap();
        }

        builder
            .bind_pipeline_compute(self.resample_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.resample_pipeline.layout().clone(),
                0,
                vec![rcx.resample_set.clone()],
            )
            .unwrap();

        unsafe {
            builder
                .dispatch([resample_extent[0].div_ceil(8), resample_extent[1].div_ceil(8), 1])
                .unwrap();
        }

        let mut info = BlitImageInfo::images(
            rcx.resample_image.clone(),
            rcx.image_views[image_index as usize].image().clone(),
        );
        info.filter = Filter::Nearest;
        builder.blit_image(info).unwrap();

        let command_buffer = builder.build().unwrap();

        let render_future =
            acquire_future.then_execute(self.queue.clone(), command_buffer).unwrap();

        let gui_future =
            rcx.gui.draw_on_image(render_future, rcx.image_views[image_index as usize].clone());

        gui_future
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
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
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();

        let (swapchain, images) = get_swapchain_images(&self.device, &surface, &window);
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
            self.memory_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            &self.render_pipeline,
            &self.resample_pipeline,
            render_extent,
            window_extent,
        );

        let gui = Gui::new(
            event_loop,
            surface,
            self.queue.clone(),
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
