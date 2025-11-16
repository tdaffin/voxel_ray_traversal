use vulkano::{
    Validated, VulkanError,
    image::view::{ImageView, ImageViewCreateInfo},
    swapchain::{SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image},
    sync::GpuFuture,
};

use crate::{
    gpu::GpuContext,
    pipelines::PipelineManager,
    rendering::{RenderContext, get_images_and_sets},
};

pub struct AcquiredFrame<F> {
    pub image_index: u32,
    pub future: F,
    pub suboptimal: bool,
}

/// Manages swapchain lifecycle: recreation on demand, image acquisition, and presentation.
pub struct SwapchainManager;

impl SwapchainManager {
    pub fn ensure_resources(
        rcx: &mut RenderContext, gpu: &GpuContext, pipelines: &PipelineManager, render_scale: f32,
    ) -> bool {
        let window_size = rcx.window.inner_size();
        if window_size.width == 0 || window_size.height == 0 {
            return false;
        }
        if rcx.recreate_swapchain {
            let images;
            (rcx.swapchain, images) = rcx
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..rcx.swapchain.create_info()
                })
                .expect("Swapchain recreation failed");
            rcx.image_views = images
                .iter()
                .map(|i| ImageView::new(i.clone(), ImageViewCreateInfo::from_image(i)).unwrap())
                .collect();
            let window_extent: [u32; 2] = window_size.into();
            let render_extent = [
                (window_extent[0] as f32 * render_scale) as u32,
                (window_extent[1] as f32 * render_scale) as u32,
            ];
            (
                rcx.render_image,
                rcx.depth_image,
                rcx.render_set,
                rcx.resample_image,
                rcx.resample_set,
            ) = get_images_and_sets(
                gpu.memory_allocator.clone(),
                gpu.descriptor_set_allocator.clone(),
                &pipelines.render,
                &pipelines.resample,
                render_extent,
                window_extent,
            );
            rcx.recreate_swapchain = false;
            true
        } else {
            false
        }
    }

    pub fn acquire(rcx: &mut RenderContext) -> Option<AcquiredFrame<Box<dyn GpuFuture>>> {
        match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
            Ok((image_index, suboptimal, future)) => Some(AcquiredFrame {
                image_index,
                future: Box::new(future) as Box<dyn GpuFuture>,
                suboptimal,
            }),
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                None
            }
            Err(e) => panic!("Failed to acquire next image: {e}"),
        }
    }

    pub fn present_and_wait<F: GpuFuture + 'static>(
        gpu: &GpuContext, rcx: &RenderContext, image_index: u32, future: F,
    ) {
        future
            .then_swapchain_present(
                gpu.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }
}
