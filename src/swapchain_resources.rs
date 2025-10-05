use vulkano::{
    image::view::{ImageView, ImageViewCreateInfo},
    swapchain::SwapchainCreateInfo,
};

use crate::gpu::GpuContext;
use crate::pipelines::PipelineManager;
use crate::rendering::{RenderContext, get_images_and_sets};

/// Recreate swapchain-dependent resources if needed. Returns true if recreation happened.
pub fn ensure_swapchain_resources(
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
        (rcx.render_image, rcx.render_set, rcx.resample_image, rcx.resample_set) =
            get_images_and_sets(
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
