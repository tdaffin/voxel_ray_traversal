use std::sync::Arc;
use vulkano::{
    buffer::BufferContents,
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, ClearColorImageInfo, CommandBufferUsage,
        PrimaryAutoCommandBuffer,
    },
    image::sampler::Filter,
    pipeline::{Pipeline, PipelineBindPoint},
};

use crate::gpu::GpuContext;
use crate::{
    camera::Camera, pipelines::PipelineManager, push_constants::PushConstantsInput,
    push_constants::build_push_constants, rendering::RenderContext, voxel_job::VoxelManager,
};

pub struct RenderOutputs {
    pub command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct ResamplePushConstants {
    display_depth: u32,
    depth_scale: f32,
}

pub fn record_frame(
    gpu: &GpuContext, pipelines: &PipelineManager, rcx: &RenderContext, voxel: &VoxelManager,
    camera: &mut Camera, render_mode: u32, image_index: u32, light_dir: [f32; 3],
    always_instant: bool, hit_back: bool, display_depth_image: bool,
) -> RenderOutputs {
    let render_extent = rcx.render_image.extent();
    let resample_extent = rcx.resample_image.extent();
    camera.extent = [render_extent[0] as f64, render_extent[1] as f64];

    let push_constants_build = build_push_constants(PushConstantsInput {
        cam_pixel_to_ray: camera.pixel_to_ray_matrix(),
        voxel,
        render_mode,
        light_dir,
        always_instant,
        hit_back,
    });
    let push_constants = push_constants_build.push_constants;
    let scene_extent = push_constants_build.scene_extent.max(1.0);
    let depth_reference = (scene_extent * 0.5).max(1.0);
    let depth_scale = 1.0 / depth_reference;

    let mut builder = AutoCommandBufferBuilder::primary(
        gpu.command_buffer_allocator.clone(),
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("Failed to create command buffer builder");

    builder.clear_color_image(ClearColorImageInfo::image(rcx.render_image.clone())).unwrap();

    builder
        .bind_pipeline_compute(pipelines.render.clone())
        .unwrap()
        .push_constants(pipelines.render.layout().clone(), 0, push_constants)
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipelines.render.layout().clone(),
            0,
            vec![rcx.render_set.clone(), voxel.voxel_set.clone()],
        )
        .unwrap();

    unsafe {
        builder.dispatch([render_extent[0].div_ceil(8), render_extent[1].div_ceil(8), 1]).unwrap();
    }

    builder
        .bind_pipeline_compute(pipelines.resample.clone())
        .unwrap()
        .push_constants(
            pipelines.resample.layout().clone(),
            0,
            ResamplePushConstants { display_depth: display_depth_image as u32, depth_scale },
        )
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipelines.resample.layout().clone(),
            0,
            vec![rcx.resample_set.clone()],
        )
        .unwrap();

    unsafe {
        builder
            .dispatch([resample_extent[0].div_ceil(8), resample_extent[1].div_ceil(8), 1])
            .unwrap();
    }

    let mut blit = BlitImageInfo::images(
        rcx.resample_image.clone(),
        rcx.image_views[image_index as usize].image().clone(),
    );
    blit.filter = Filter::Nearest;
    builder.blit_image(blit).unwrap();

    let command_buffer = builder.build().unwrap();
    RenderOutputs { command_buffer }
}
