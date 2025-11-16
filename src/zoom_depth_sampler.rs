use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
    allocator::StandardCommandBufferAllocator,
};
use vulkano::device::Queue;
use vulkano::image::{Image, ImageAspects, ImageSubresourceLayers};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::{self, GpuFuture};

const SAMPLE_EDGE: u32 = 9;
const SAMPLE_CAPACITY: usize = (SAMPLE_EDGE * SAMPLE_EDGE) as usize;
const SAMPLE_OFFSETS: &[(i32, i32)] = &[
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
    (2, 0),
    (-2, 0),
    (0, 2),
    (0, -2),
];

pub struct ZoomDepthSampler {
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    staging: Subbuffer<[f32]>,
}

impl ZoomDepthSampler {
    pub fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>, queue: Arc<Queue>,
    ) -> Self {
        let staging: Subbuffer<[f32]> = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::TRANSFER_DST, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            SAMPLE_CAPACITY as u64,
        )
        .expect("failed to allocate depth sampling buffer");

        Self { command_buffer_allocator, queue, staging }
    }

    pub fn sample(
        &mut self, depth_image: Arc<Image>, render_extent: [u32; 2], focus_px: [u32; 2],
    ) -> Option<f64> {
        let (width, height) = (render_extent[0], render_extent[1]);
        if width == 0 || height == 0 {
            return None;
        }
        let block_w = width.min(SAMPLE_EDGE).max(1);
        let block_h = height.min(SAMPLE_EDGE).max(1);

        let clamped_focus = [focus_px[0].min(width - 1), focus_px[1].min(height - 1)];

        let base_x = clamp_block_origin(clamped_focus[0], width, block_w);
        let base_y = clamp_block_origin(clamped_focus[1], height, block_h);
        let center_x = (clamped_focus[0] - base_x) as i32;
        let center_y = (clamped_focus[1] - base_y) as i32;

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("failed to create command buffer builder for zoom sampling");

        let mut copy_info =
            CopyImageToBufferInfo::image_buffer(depth_image.clone(), self.staging.clone());
        let region = &mut copy_info.regions[0];
        region.image_subresource = ImageSubresourceLayers {
            aspects: ImageAspects::COLOR,
            mip_level: 0,
            array_layers: 0..1,
        };
        region.image_offset = [base_x, base_y, 0];
        region.image_extent = [block_w, block_h, 1];
        region.buffer_offset = 0;
        region.buffer_row_length = block_w;
        region.buffer_image_height = block_h;
        builder.copy_image_to_buffer(copy_info).unwrap();

        let command_buffer = builder.build().unwrap();
        let future = sync::now(self.queue.device().clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let block_w_usize = block_w as usize;
        let block_h_usize = block_h as usize;
        let data_guard = self.staging.read().unwrap();
        let data = &data_guard[..block_w_usize * block_h_usize];

        for &(ox, oy) in SAMPLE_OFFSETS.iter() {
            let sx = center_x + ox;
            let sy = center_y + oy;
            if sx < 0 || sy < 0 || sx >= block_w as i32 || sy >= block_h as i32 {
                continue;
            }
            let idx = sy as usize * block_w_usize + sx as usize;
            let depth = data[idx];
            if depth.is_finite() && depth < f32::MAX * 0.5 {
                return Some(depth.max(0.0) as f64);
            }
        }
        None
    }
}

fn clamp_block_origin(coord: u32, total: u32, block: u32) -> u32 {
    if total <= block {
        0
    } else {
        let half = block / 2;
        coord.saturating_sub(half).min(total - block)
    }
}
