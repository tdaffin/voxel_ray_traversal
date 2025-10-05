use vulkano::{
    Validated, VulkanError,
    swapchain::{SwapchainPresentInfo, acquire_next_image},
    sync::GpuFuture,
};

use crate::{gpu::GpuContext, rendering::RenderContext};

pub struct AcquireResult<F> {
    pub image_index: u32,
    pub future: F,
    pub suboptimal: bool,
}

/// Attempts to acquire the next swapchain image. Returns None if swapchain is out of date (caller should recreate and skip frame).
pub fn acquire_image(rcx: &mut RenderContext) -> Option<AcquireResult<Box<dyn GpuFuture>>> {
    match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
        Ok((image_index, suboptimal, future)) => Some(AcquireResult {
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

/// Presents the final image and waits for completion.
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
