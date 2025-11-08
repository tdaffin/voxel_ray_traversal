use std::{
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
    pipelines::PipelineManager,
    push_constants::{PushConstantsInput, build_push_constants},
    render_mode::RenderMode,
    rendering::get_render_image,
    voxel_facade::VoxelSystem,
};

pub struct RenderSnapshot {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl RenderSnapshot {
    pub fn save_png(&self, path: &Path) -> ImageResult<()> {
        let img = image::RgbaImage::from_raw(self.width, self.height, self.pixels.clone())
            .expect("failed to convert snapshot to image");
        img.save(path)
    }
}

pub fn render_offscreen_snapshot(
    width: u32, height: u32, render_mode: RenderMode,
) -> RenderSnapshot {
    let gpu = GpuContext::headless();
    let shaders_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let pipelines = PipelineManager::new(gpu.device.clone(), &shaders_dir);

    let mut voxel = VoxelSystem::new(
        24,
        gpu.memory_allocator.clone(),
        gpu.descriptor_set_allocator.clone(),
        gpu.command_buffer_allocator.clone(),
        gpu.queue.clone(),
        &pipelines.render,
    );

    wait_for_voxel_jobs(
        &mut voxel,
        &pipelines.render,
        gpu.descriptor_set_allocator.clone(),
        gpu.memory_allocator.clone(),
        gpu.command_buffer_allocator.clone(),
        gpu.queue.clone(),
    );

    let mut camera = {
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
    let layout = pipelines.render.layout().set_layouts()[0].clone();
    let render_set = DescriptorSet::new(
        gpu.descriptor_set_allocator.clone(),
        layout,
        [WriteDescriptorSet::image_view(0, render_view.clone())],
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
    });

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
    let timeout = Duration::from_secs(30);
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

    #[test]
    fn offscreen_render_matches_reference() {
        let snapshot = render_offscreen_snapshot(256, 256, RenderMode::Shade);
        let out_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference");
        std::fs::create_dir_all(&out_dir).expect("failed to create reference directory");
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
}
