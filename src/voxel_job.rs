use std::sync::mpsc::{self, Receiver};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::{DescriptorSet, allocator::StandardDescriptorSetAllocator};
use vulkano::device::Queue;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::StandardMemoryAllocator;

use crate::{
    hot_reload::HotReloadComputePipeline,
    model::Model,
    voxel::{build_voxel_descriptor_set, create_empty_voxel_placeholder, create_voxel_image_view},
    voxelize,
};

#[derive(Debug)]
pub enum VoxelJobMessage {
    Finished { generation: u64, index: usize, data: Vec<u128> },
    Progress { generation: u64, index: usize, done: usize, total: usize },
    Cancelled { generation: u64 },
}

/// Manages voxel grid generation jobs, descriptor set, and progress.
pub struct VoxelManager {
    pub voxel_set: Arc<DescriptorSet>,
    pub voxel_resolution: u32,
    pub grid_resolutions: Vec<u32>,
    pub future_grid_resolutions: Vec<u32>,
    pub active_voxel_grids: u32,

    pub voxel_views: Vec<Option<Arc<ImageView>>>,
    pub voxel_pending: Vec<bool>,
    pub voxel_progress: Vec<(usize, usize)>,

    placeholder_view: Arc<ImageView>,
    voxel_result_rx: Receiver<VoxelJobMessage>,
    voxel_generation: u64,
    cancel_requested: bool,
    voxel_cancel_flag: Arc<AtomicBool>,
}

impl VoxelManager {
    pub fn new(
        initial_resolution: u32, memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>, queue: Arc<Queue>,
        render_pipeline: &HotReloadComputePipeline,
    ) -> Self {
        let grid_resolutions = vec![initial_resolution; Model::ALL.len()];
        let future_grid_resolutions = grid_resolutions.clone();
        let placeholder_view = create_empty_voxel_placeholder(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            initial_resolution,
        );
        let voxel_set = build_voxel_descriptor_set(
            descriptor_set_allocator.clone(),
            render_pipeline,
            std::slice::from_ref(&placeholder_view),
        );
        let (tx, rx) = mpsc::channel();
        let voxel_generation = 1u64;
        for (idx, model) in Model::ALL.iter().enumerate() {
            let txc = tx.clone();
            let path = model.path().as_ref().to_path_buf();
            let res = grid_resolutions[idx];
            let generation_id = voxel_generation;
            thread::spawn(move || {
                let vox = voxelize::ply_to_voxels(path, res);
                let _ = txc.send(VoxelJobMessage::Finished {
                    generation: generation_id,
                    index: idx,
                    data: vox,
                });
            });
        }
        drop(tx);
        let voxel_views = vec![None; Model::ALL.len()];
        let voxel_pending = vec![true; Model::ALL.len()];
        let voxel_progress = vec![(0, 0); Model::ALL.len()];
        let voxel_cancel_flag = Arc::new(AtomicBool::new(false));
        let active_voxel_grids = Model::ALL.len() as u32;
        Self {
            voxel_set,
            voxel_resolution: initial_resolution,
            grid_resolutions,
            future_grid_resolutions,
            active_voxel_grids,
            voxel_views,
            voxel_pending,
            voxel_progress,
            placeholder_view,
            voxel_result_rx: rx,
            voxel_generation,
            cancel_requested: false,
            voxel_cancel_flag,
        }
    }

    pub fn poll(
        &mut self, descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        render_pipeline: &HotReloadComputePipeline, memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>, queue: Arc<Queue>,
    ) {
        while let Ok(msg) = self.voxel_result_rx.try_recv() {
            match msg {
                VoxelJobMessage::Progress { generation, index, done, total } => {
                    if generation != self.voxel_generation {
                        continue;
                    }
                    if index < self.voxel_progress.len() {
                        self.voxel_progress[index] = (done, total);
                    }
                }
                VoxelJobMessage::Finished { generation, index, data } => {
                    if generation != self.voxel_generation || self.cancel_requested {
                        continue;
                    }
                    if index < self.voxel_views.len() {
                        let view = create_voxel_image_view(
                            memory_allocator.clone(),
                            command_buffer_allocator.clone(),
                            queue.clone(),
                            data,
                            self.voxel_resolution,
                        );
                        self.voxel_views[index] = Some(view);
                        self.voxel_pending[index] = false;
                    }
                }
                VoxelJobMessage::Cancelled { generation } => {
                    if generation != self.voxel_generation {
                        continue;
                    }
                    if self.cancel_requested {
                        self.voxel_pending.fill(false);
                    }
                }
            }
            // rebuild contiguous descriptor set
            let mut ready: Vec<Arc<ImageView>> = Vec::new();
            for opt in &self.voxel_views {
                if let Some(v) = opt {
                    ready.push(v.clone());
                } else {
                    break;
                }
            }
            self.active_voxel_grids = ready.len() as u32;
            if self.active_voxel_grids > 0 {
                self.voxel_set = build_voxel_descriptor_set(
                    descriptor_set_allocator.clone(),
                    render_pipeline,
                    &ready,
                );
            }
        }
    }

    pub fn cancel(&mut self) {
        self.cancel_requested = true;
        self.voxel_cancel_flag.store(true, Ordering::Relaxed);
    }

    pub fn regenerate(
        &mut self, descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>, queue: Arc<Queue>,
        render_pipeline: &HotReloadComputePipeline, placeholder_resolution: u32,
    ) {
        // apply future resolutions
        for (i, r) in self.future_grid_resolutions.clone().into_iter().enumerate() {
            if i < self.grid_resolutions.len() {
                self.grid_resolutions[i] = r.div_ceil(8) * 8;
            }
        }
        if let Some(maxr) = self.grid_resolutions.iter().copied().max() {
            self.voxel_resolution = maxr;
        }
        self.placeholder_view = create_empty_voxel_placeholder(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            placeholder_resolution,
        );
        self.start_background_voxelization(descriptor_set_allocator, render_pipeline);
    }

    fn start_background_voxelization(
        &mut self, descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        render_pipeline: &HotReloadComputePipeline,
    ) {
        for v in &mut self.voxel_views {
            *v = None;
        }
        self.active_voxel_grids = 0;
        self.voxel_generation = self.voxel_generation.wrapping_add(1);
        self.voxel_pending.fill(true);
        self.cancel_requested = false;
        self.voxel_cancel_flag.store(false, Ordering::Relaxed);
        for p in &mut self.voxel_progress {
            *p = (0, 0);
        }
        self.voxel_set = build_voxel_descriptor_set(
            descriptor_set_allocator.clone(),
            render_pipeline,
            std::slice::from_ref(&self.placeholder_view),
        );
        let (tx, rx) = mpsc::channel();
        self.voxel_result_rx = rx;
        let generation_id = self.voxel_generation;
        let cancel_flag = self.voxel_cancel_flag.clone();
        for (idx, model) in Model::ALL.iter().enumerate() {
            let txc = tx.clone();
            let path = model.path().as_ref().to_path_buf();
            let gen_thread = generation_id;
            let cancel_local = cancel_flag.clone();
            let res_for_grid = self.grid_resolutions[idx];
            thread::spawn(move || {
                use crate::voxelize::{VoxelProgressCallbacks, ply_to_voxels_with_progress};
                let cancelled = cancel_local.clone();
                let prog_cb = VoxelProgressCallbacks {
                    cancelled: &cancelled,
                    progress: Some(Box::new({
                        let txp = txc.clone();
                        move |done, total| {
                            let _ = txp.send(VoxelJobMessage::Progress {
                                generation: gen_thread,
                                index: idx,
                                done,
                                total,
                            });
                        }
                    })),
                };
                if let Some(vox) = ply_to_voxels_with_progress(path, res_for_grid, prog_cb) {
                    if !cancelled.load(Ordering::Relaxed) {
                        let _ = txc.send(VoxelJobMessage::Finished {
                            generation: gen_thread,
                            index: idx,
                            data: vox,
                        });
                    } else {
                        let _ = txc.send(VoxelJobMessage::Cancelled { generation: gen_thread });
                    }
                } else {
                    let _ = txc.send(VoxelJobMessage::Cancelled { generation: gen_thread });
                }
            });
        }
        drop(tx);
    }
}
