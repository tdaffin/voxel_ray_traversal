use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::device::Device;
use vulkano::device::Queue;

use crate::{
    benchmark::{BenchmarkContext, run},
    camera::Camera,
    pipelines::PipelineManager,
    rendering::RenderContext,
    voxel_job::VoxelManager,
};

pub fn run_bench_if_requested(
    trigger: bool, frames: u32, camera: &Camera, voxel: &VoxelManager, render_mode: u32,
    always_instant: bool, pipelines: &PipelineManager, queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>, rcx: &mut RenderContext,
    device: Arc<Device>,
) {
    if !trigger {
        return;
    }

    let outcome = run(&mut BenchmarkContext {
        frames,
        camera,
        voxel,
        render_mode,
        always_instant,
        branching_pipeline: pipelines.render.clone(),
        branchless_pipeline: pipelines.render_branchless.clone(),
        queue: queue.clone(),
        command_buffer_allocator: command_buffer_allocator.clone(),
        rcx,
        device: device.clone(),
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
