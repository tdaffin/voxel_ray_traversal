use std::{
    collections::HashSet,
    path::Path,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use image::ImageResult;
use nalgebra::Vector3;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::allocator::StandardCommandBufferAllocator,
    command_buffer::{
        AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyImageToBufferInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
};

use crate::{
    camera::Camera,
    gpu::GpuContext,
    hot_reload::HotReloadComputePipeline,
    model_discovery::{DiscoveredModel, discover_models},
    pipelines::PipelineManager,
    push_constants::{PushConstantsInput, build_push_constants},
    render_mode::RenderMode,
    rendering::{get_depth_image, get_render_image},
    voxel_facade::VoxelSystem,
};

#[cfg_attr(not(test), allow(dead_code))]
pub struct RenderSnapshot {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
pub enum SnapshotModelSelection<'a> {
    All,
    Named(&'a [&'a str]),
}

impl<'a> Default for SnapshotModelSelection<'a> {
    fn default() -> Self {
        SnapshotModelSelection::All
    }
}

#[derive(Clone, Debug)]
pub struct RenderSnapshotOptions<'a> {
    pub model_selection: SnapshotModelSelection<'a>,
    pub camera: Option<Camera>,
}

impl<'a> Default for RenderSnapshotOptions<'a> {
    fn default() -> Self {
        Self { model_selection: SnapshotModelSelection::All, camera: None }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl RenderSnapshot {
    pub fn save_png(&self, path: &Path) -> ImageResult<()> {
        let img = image::RgbaImage::from_raw(self.width, self.height, self.pixels.clone())
            .expect("failed to convert snapshot to image");
        img.save(path)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn render_offscreen_snapshot(
    width: u32, height: u32, render_mode: RenderMode, options: RenderSnapshotOptions<'_>,
) -> RenderSnapshot {
    let gpu = GpuContext::headless();
    let shaders_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let pipelines = PipelineManager::new(gpu.device.clone(), &shaders_dir);

    let models_override: Option<Vec<DiscoveredModel>> = match options.model_selection {
        SnapshotModelSelection::All => None,
        SnapshotModelSelection::Named(names) => {
            if names.is_empty() {
                Some(Vec::new())
            } else {
                let wanted: HashSet<String> =
                    names.iter().map(|name| name.to_ascii_lowercase()).collect();
                let filtered: Vec<DiscoveredModel> = discover_models()
                    .into_iter()
                    .filter(|model| {
                        let stem_lower = model.name.to_ascii_lowercase();
                        if wanted.contains(&stem_lower) {
                            return true;
                        }
                        model
                            .path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .map(|s| wanted.contains(&s.to_ascii_lowercase()))
                            .unwrap_or(false)
                    })
                    .collect();

                if filtered.is_empty() {
                    eprintln!("[offscreen] snapshot model filter matched zero models: {:?}", names);
                }

                Some(filtered)
            }
        }
    };

    let mut voxel = VoxelSystem::new(
        24,
        gpu.memory_allocator.clone(),
        gpu.descriptor_set_allocator.clone(),
        gpu.command_buffer_allocator.clone(),
        gpu.queue.clone(),
        &pipelines.render,
        models_override,
    );

    wait_for_voxel_jobs(
        &mut voxel,
        &pipelines.render,
        gpu.descriptor_set_allocator.clone(),
        gpu.memory_allocator.clone(),
        gpu.command_buffer_allocator.clone(),
        gpu.queue.clone(),
    );

    let mut camera = if let Some(mut cam) = options.camera.clone() {
        cam.extent = [width as f64, height as f64];
        cam
    } else {
        let target = Vector3::new(-0.34, -0.28, -0.45);
        let cam_pos = target + Vector3::new(-0.35, 0.45, 0.45);
        let mut camera =
            Camera::new(cam_pos, Vector3::zeros(), [width as f64, height as f64], 35.0);
        camera.look_at(target);
        camera
    };
    camera.extent = [width as f64, height as f64];

    let (render_image, render_view) =
        get_render_image(gpu.memory_allocator.clone(), [width, height]);
    let (_depth_image, depth_view) = get_depth_image(gpu.memory_allocator.clone(), [width, height]);
    let layout = pipelines.render.layout().set_layouts()[0].clone();
    let render_set = DescriptorSet::new(
        gpu.descriptor_set_allocator.clone(),
        layout,
        [
            WriteDescriptorSet::image_view(0, render_view.clone()),
            WriteDescriptorSet::image_view(1, depth_view.clone()),
        ],
        [],
    )
    .expect("failed to create render descriptor set");

    let push_constants = build_push_constants(PushConstantsInput {
        cam_pixel_to_ray: camera.pixel_to_ray_matrix(),
        voxel: &voxel.manager,
        render_mode: render_mode as u32,
        light_dir: [0.5, 0.8, 0.3],
        always_instant: false,
        hit_back: false,
    })
    .push_constants;

    let pixel_count = (width * height) as usize;
    let staging: Subbuffer<[u8]> = Buffer::new_slice(
        gpu.memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (pixel_count * 4) as u64,
    )
    .expect("failed to allocate staging buffer");

    let mut builder = AutoCommandBufferBuilder::primary(
        gpu.command_buffer_allocator.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("failed to create command buffer builder");

    builder.clear_color_image(ClearColorImageInfo::image(render_image.clone())).unwrap();
    builder
        .bind_pipeline_compute(pipelines.render.clone())
        .unwrap()
        .push_constants(pipelines.render.layout().clone(), 0, push_constants)
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipelines.render.layout().clone(),
            0,
            vec![render_set.clone(), voxel.manager.voxel_set.clone()],
        )
        .unwrap();

    unsafe {
        builder.dispatch([width.div_ceil(8), height.div_ceil(8), 1]).unwrap();
    }

    builder
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            render_image.clone(),
            staging.clone(),
        ))
        .unwrap();

    let command_buffer = builder.build().unwrap();
    let future = sync::now(gpu.device.clone())
        .then_execute(gpu.queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();

    let read_guard = staging.read().unwrap();
    let pixels = read_guard.to_vec();

    RenderSnapshot { width, height, pixels }
}

fn wait_for_voxel_jobs(
    voxel: &mut VoxelSystem, render_pipeline: &HotReloadComputePipeline,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>, queue: Arc<Queue>,
) {
    let timeout = Duration::from_secs(90);
    let start = Instant::now();
    loop {
        voxel.manager.poll(
            descriptor_set_allocator.clone(),
            render_pipeline,
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            0.0,
        );

        if voxel.manager.voxel_pending.iter().all(|pending| !pending) {
            break;
        }

        if start.elapsed() > timeout {
            panic!("voxel generation timed out");
        }

        thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    static FULL_SNAPSHOT: OnceLock<RenderSnapshot> = OnceLock::new();

    fn full_snapshot() -> &'static RenderSnapshot {
        FULL_SNAPSHOT.get_or_init(|| make_full_snapshot(RenderMode::Shade))
    }

    fn make_full_snapshot(render_mode: RenderMode) -> RenderSnapshot {
        let target = Vector3::new(-0.34, -0.28, -0.45);
        let cam_pos = target + Vector3::new(-0.5, 0.5, 0.5);
        let mut camera = Camera::new(cam_pos, Vector3::zeros(), [256f64, 256f64], 35.0);
        camera.look_at(target);
        render_offscreen_snapshot(
            256,
            256,
            render_mode,
            RenderSnapshotOptions {
                model_selection: SnapshotModelSelection::Named(&[
                    "teapot", "chr_cat", "chr_bow", "chr_fox",
                ]),
                camera: Some(camera),
            },
        )
        //RenderSnapshotOptions::default()
    }

    static EMPTY_SNAPSHOT: OnceLock<RenderSnapshot> = OnceLock::new();

    fn empty_snapshot() -> &'static RenderSnapshot {
        EMPTY_SNAPSHOT.get_or_init(|| {
            render_offscreen_snapshot(
                256,
                256,
                RenderMode::Shade,
                RenderSnapshotOptions {
                    model_selection: SnapshotModelSelection::Named(&[]),
                    camera: None,
                },
            )
        })
    }

    fn out_dir() -> std::path::PathBuf {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference");
        std::fs::create_dir_all(&dir).expect("failed to create reference directory");
        return dir;
    }

    #[test]
    fn offscreen_render_matches_empty_reference() {
        let snapshot = empty_snapshot();
        let out_dir = out_dir();
        let baseline_path = out_dir.join("empty_reference.png");
        let update_baseline = std::env::var("VOXEL_RENDER_UPDATE").is_ok();

        if update_baseline || !baseline_path.exists() {
            snapshot.save_png(&baseline_path).expect("failed to write baseline image");
            if !update_baseline {
                panic!(
                    "Reference image missing; saved snapshot to {}. Rerun with VOXEL_RENDER_UPDATE=1 on a known-good branch to populate the baseline.",
                    baseline_path.display()
                );
            }
        }

        let expected = image::open(&baseline_path)
            .expect("failed to load baseline image")
            .to_rgba8()
            .into_raw();

        if snapshot.pixels != expected {
            let actual_path = out_dir.join("empty_reference.actual.png");
            snapshot.save_png(&actual_path).expect("failed to write actual render output");
            panic!(
                "Rendered snapshot diverged. Examine {} and rerun with VOXEL_RENDER_UPDATE=1 if the change is expected.",
                actual_path.display()
            );
        }
    }

    #[test]
    fn offscreen_render_matches_reference() {
        let snapshot = full_snapshot();
        let out_dir = out_dir();
        let baseline_path = out_dir.join("shade_reference.png");
        let update_baseline = std::env::var("VOXEL_RENDER_UPDATE").is_ok();

        if update_baseline || !baseline_path.exists() {
            snapshot.save_png(&baseline_path).expect("failed to write baseline image");
            if !update_baseline {
                panic!(
                    "Reference image missing; saved snapshot to {}. Rerun with VOXEL_RENDER_UPDATE=1 on a known-good branch to populate the baseline.",
                    baseline_path.display()
                );
            }
        }

        let expected = image::open(&baseline_path)
            .expect("failed to load baseline image")
            .to_rgba8()
            .into_raw();

        if snapshot.pixels != expected {
            let actual_path = out_dir.join("shade_reference.actual.png");
            snapshot.save_png(&actual_path).expect("failed to write actual render output");
            panic!(
                "Rendered snapshot diverged. Examine {} and rerun with VOXEL_RENDER_UPDATE=1 if the change is expected.",
                actual_path.display()
            );
        }
    }

    #[test]
    fn verify_no_transparent() {
        // 220, 100, 30x30
        let snapshot = full_snapshot();
        let mut num_transparent = 0;
        for y in 140..=170 {
            for x in 220..=250 {
                let idx = (y * snapshot.width as usize + x) * 4;
                let px = &snapshot.pixels[idx..idx + 4];
                let mut sum = 0u32;
                for i in 0..4 {
                    sum += px[i] as u32;
                    //snapshot.pixels[idx + i] = 0;
                }
                if sum == 0 {
                    num_transparent += 1;
                }
            }
        }
        if (num_transparent > 0) {
            let out_dir = out_dir();
            let actual_path = out_dir.join("no_transparent.actual.png");
            snapshot.save_png(&actual_path).expect("failed to write actual render output");
        }
        assert_eq!(
            num_transparent, 0,
            "Found {} transparent pixels in the test area",
            num_transparent
        );
    }

    #[ignore = "pending investigation"]
    #[test]
    fn verify_all_green() {
        // 84, 105, 11x6
        //let snapshot = full_snapshot();
        let mut snapshot = make_full_snapshot(RenderMode::Debug);
        let mut num_not_green = 0;
        for y in 165..=175 {
            for x in 84..=95 {
                let idx = (y * snapshot.width as usize + x) * 4;
                let px = &snapshot.pixels[idx..idx + 4];
                let px32 = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
                if px32 != 0xFF408C33 {
                    num_not_green += 1;
                    println!("Pixel at ({},{}) is not green: #{:08X}", x, y, px32);
                }
                for i in 0..4 {
                    //snapshot.pixels[idx + i] = 0;
                }
            }
        }
        if num_not_green > 0 {
            let out_dir = out_dir();
            let actual_path = out_dir.join("all_green.actual.png");
            snapshot.save_png(&actual_path).expect("failed to write actual render output");
        }
        assert_eq!(num_not_green, 0, "Found {} non-green pixels in the test area", num_not_green);
    }
}
