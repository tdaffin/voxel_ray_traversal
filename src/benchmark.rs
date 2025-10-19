use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::camera::Camera;
use crate::push_constants::{PushConstantsInput, build_push_constants};
use crate::voxel_job::VoxelManager;

use vulkano::{
    Validated,
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, ClearColorImageInfo, CommandBufferUsage,
    },
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
    swapchain::{SwapchainPresentInfo, acquire_next_image},
    sync::GpuFuture,
};

use crate::rendering::RenderContext;

pub struct BenchmarkOutcome {
    pub frames: u32,
    pub branching_cpu_avg: Duration,
    pub branching_gpu_avg_ns: Option<u128>,
    pub branchless_cpu_avg: Duration,
    pub branchless_gpu_avg_ns: Option<u128>,
}

pub struct BenchmarkContext<'a> {
    pub frames: u32,
    pub camera: &'a Camera,
    pub voxel: &'a VoxelManager,
    pub render_mode: u32,
    pub always_instant: bool,
    pub branching_pipeline: Arc<ComputePipeline>,
    pub branchless_pipeline: Arc<ComputePipeline>,
    pub queue: Arc<vulkano::device::Queue>,
    pub command_buffer_allocator:
        Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator>,
    pub rcx: &'a mut RenderContext,
    pub device: Arc<vulkano::device::Device>,
}

pub fn run(ctx: &mut BenchmarkContext) -> BenchmarkOutcome {
    let frames = ctx.frames; // frames each variant

    let run_variant = |pipe: Arc<ComputePipeline>,
                       rcx: &mut RenderContext|
     -> (Duration, Option<u128>) {
        let mut total_cpu = Duration::ZERO;
        let mut total_gpu_ns_accum: f64 = 0.0;
        let timestamp_period = ctx.device.physical_device().properties().timestamp_period; // ns per tick
        let supports_timestamps = ctx.device.physical_device().queue_family_properties()
            [ctx.queue.queue_family_index() as usize]
            .timestamp_valid_bits
            .is_some();
        let query_pool = if supports_timestamps {
            Some(
                QueryPool::new(
                    ctx.device.clone(),
                    QueryPoolCreateInfo {
                        query_count: frames * 2,
                        ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
                    },
                )
                .unwrap(),
            )
        } else {
            None
        };

        for f in 0..frames {
            let (image_index, suboptimal, acquire_future) =
                match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
                    Ok(r) => r,
                    Err(_) => break,
                };
            if suboptimal {
                rcx.recreate_swapchain = true;
            }
            let render_extent = rcx.render_image.extent();
            let push_constants = build_push_constants(PushConstantsInput {
                cam_pixel_to_ray: ctx.camera.pixel_to_ray_matrix(),
                voxel: ctx.voxel,
                render_mode: ctx.render_mode,
                light_dir: [0.5, 0.8, 0.3],
                always_instant: ctx.always_instant,
            });
            let mut builder = AutoCommandBufferBuilder::primary(
                ctx.command_buffer_allocator.clone(),
                ctx.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder
                .clear_color_image(ClearColorImageInfo::image(rcx.render_image.clone()))
                .unwrap();
            if let Some(qp) = &query_pool {
                unsafe {
                    builder
                        .write_timestamp(qp.clone(), f * 2, vulkano::sync::PipelineStage::TopOfPipe)
                        .unwrap();
                }
            }
            builder
                .bind_pipeline_compute(pipe.clone())
                .unwrap()
                .push_constants(pipe.layout().clone(), 0, push_constants)
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipe.layout().clone(),
                    0,
                    vec![rcx.render_set.clone(), ctx.voxel.voxel_set.clone()],
                )
                .unwrap();
            unsafe {
                builder
                    .dispatch([render_extent[0].div_ceil(8), render_extent[1].div_ceil(8), 1])
                    .unwrap();
            }
            if let Some(qp) = &query_pool {
                unsafe {
                    builder
                        .write_timestamp(
                            qp.clone(),
                            f * 2 + 1,
                            vulkano::sync::PipelineStage::BottomOfPipe,
                        )
                        .unwrap();
                }
            }
            let mut info = BlitImageInfo::images(
                rcx.render_image.clone(),
                rcx.image_views[image_index as usize].image().clone(),
            );
            info.filter = vulkano::image::sampler::Filter::Nearest;
            builder.blit_image(info).unwrap();
            let command_buffer = builder.build().unwrap();
            let start = Instant::now();
            let future = acquire_future
                .then_execute(ctx.queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(
                    ctx.queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index),
                )
                .then_signal_fence_and_flush()
                .unwrap();
            future.wait(None).unwrap();
            total_cpu += start.elapsed();
        }
        let avg_gpu_ns_opt = if let Some(qp) = &query_pool {
            if supports_timestamps {
                let mut data: Vec<u64> = vec![0; (frames * 2) as usize];
                qp.get_results(0..frames * 2, &mut data, QueryResultFlags::WAIT).unwrap();
                for f in 0..frames {
                    let start = data[(f * 2) as usize];
                    let end = data[(f * 2 + 1) as usize];
                    if end > start {
                        let ticks = (end - start) as f64;
                        total_gpu_ns_accum += ticks * timestamp_period as f64;
                    }
                }
                if total_gpu_ns_accum > 0.0 {
                    Some((total_gpu_ns_accum / frames as f64) as u128)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        (total_cpu / frames, avg_gpu_ns_opt)
    };

    let (branching_cpu, branching_gpu) = run_variant(ctx.branching_pipeline.clone(), ctx.rcx);
    let (branchless_cpu, branchless_gpu) = run_variant(ctx.branchless_pipeline.clone(), ctx.rcx);

    BenchmarkOutcome {
        frames,
        branching_cpu_avg: branching_cpu,
        branching_gpu_avg_ns: branching_gpu,
        branchless_cpu_avg: branchless_cpu,
        branchless_gpu_avg_ns: branchless_gpu,
    }
}
