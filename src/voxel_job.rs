use std::sync::mpsc::{self, Receiver};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::{DescriptorSet, allocator::StandardDescriptorSetAllocator};
use vulkano::device::Queue;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::StandardMemoryAllocator;

use crate::{
    hot_reload::HotReloadComputePipeline,
    model_discovery::{DiscoveredModel, discover_models},
    voxel::{
        GridInfo, build_voxel_descriptor_set, create_empty_voxel_placeholder,
        create_grid_info_buffer, create_voxel_image_view,
    },
    voxelize,
};

#[derive(Debug)]
pub enum VoxelJobMessage {
    Finished {
        generation: u64,
        index: usize,
        data: Vec<u128>,
        colors: Vec<u8>,
        resolution: u32,
        dim_x: u32,
        dim_y: u32,
        dim_z: u32,
    },
    PaletteSlice {
        generation: u64,
        base: u8,
        colors: Vec<[f32; 4]>,
    },
    Progress {
        generation: u64,
        index: usize,
        done: usize,
        total: usize,
    },
    Cancelled {
        generation: u64,
    },
}

/// Manages voxel grid generation jobs, descriptor set, and progress.
pub struct VoxelManager {
    pub models: Vec<DiscoveredModel>,
    pub voxel_set: Arc<DescriptorSet>,
    pub voxel_resolution: u32,
    pub grid_resolutions: Vec<u32>,
    pub grid_dims: Vec<(u32, u32, u32)>,
    pub future_grid_resolutions: Vec<u32>,
    pub active_voxel_grids: u32,

    pub voxel_views: Vec<Option<Arc<ImageView>>>,
    pub color_index_views: Vec<Option<Arc<ImageView>>>,
    pub voxel_pending: Vec<bool>,
    pub voxel_progress: Vec<(usize, usize)>,

    placeholder_view: Arc<ImageView>,
    placeholder_color_view: Arc<ImageView>,
    palette_buffer: Subbuffer<[[f32; 4]]>,
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
        let models = discover_models();
        let model_count = models.len().max(1); // avoid div by zero in palette math
        let grid_resolutions = vec![initial_resolution; model_count];
        let grid_dims =
            vec![(initial_resolution, initial_resolution, initial_resolution); model_count];
        let future_grid_resolutions = grid_resolutions.clone();
        let placeholder_view = create_empty_voxel_placeholder(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            initial_resolution,
        );
        let placeholder_color_view = crate::voxel::create_empty_color_index_placeholder(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            initial_resolution,
        );
        // Build default palette (simple distinct hues)
        let mut palette: Vec<[f32; 4]> = Vec::new();
        for i in 0..256u32 {
            let h = (i as f32) / 256.0;
            // HSV to RGB (s=0.85,v=1.0) quick approximation
            let s = 0.85;
            let v = 1.0;
            let k = |n: f32| ((n + h * 6.0) % 6.0).clamp(0.0, 6.0);
            let f = |n: f32| v - v * s * f32::max(0.0, f32::min(f32::min(k(n), 4.0 - k(n)), 1.0));
            let r = f(5.0);
            let g = f(3.0);
            let b = f(1.0);
            palette.push([r, g, b, 1.0]);
        }
        let palette_buffer =
            crate::voxel::create_palette_buffer(memory_allocator.clone(), &palette);
        // Initial grid info (single placeholder)
        let gi = [GridInfo {
            resolution: initial_resolution,
            origin_x: 0.0,
            origin_y: 0.0,
            dim_x: initial_resolution,
            dim_y: initial_resolution,
            dim_z: initial_resolution,
            _pad0: 0,
        }];
        let grid_info_buffer = create_grid_info_buffer(memory_allocator.clone(), &gi);
        let voxel_set = build_voxel_descriptor_set(
            descriptor_set_allocator.clone(),
            render_pipeline,
            std::slice::from_ref(&placeholder_view),
            std::slice::from_ref(&placeholder_color_view),
            palette_buffer.clone(),
            grid_info_buffer.clone(),
        );
        let (tx, rx) = mpsc::channel();
        let voxel_generation = 1u64;
        for (idx, model) in models.iter().enumerate() {
            let txc = tx.clone();
            let path = model.path.clone();
            let res = grid_resolutions[idx];
            let generation_id = voxel_generation;
            let base_palette_index = (idx as u32 * (256 / model_count as u32)) as u8;
            // Each model gets a contiguous 16-color sub-range (low nibble variation added during voxel write).
            thread::spawn(move || {
                let palette_span = (256 / model_count as u32) as u8; // subrange reserved per model
                let (vox, colors, palette_opt, native_res_opt, dims_opt) =
                    if path.extension().and_then(|e| e.to_str()) == Some("vox") {
                        if !path.exists() {
                            eprintln!("[voxel] .vox file missing: {}", path.display());
                            (
                                vec![0u128; (res as usize).pow(3) / 128],
                                vec![0u8; (res as usize).pow(3)],
                                None,
                                None,
                                None,
                            )
                        } else if let Some(result) = crate::voxelize_vox::vox_to_voxels(
                            &path,
                            None,
                            base_palette_index,
                            palette_span,
                        ) {
                            let v = result.voxels;
                            let c = result.colors;
                            let p = result.palette;
                            let used_res = result.used_resolution;
                            let dims_tuple = (result.dim_x, result.dim_y, result.dim_z);
                            if v.iter().all(|&u| u == 0) {
                                eprintln!(
                                    "[voxel] WARNING: .vox produced empty voxel set: {}",
                                    path.display()
                                );
                            }
                            (v, c, Some(p), Some(used_res), Some(dims_tuple))
                        } else {
                            eprintln!("[voxel] Failed to parse .vox file: {}", path.display());
                            (
                                vec![0u128; (res as usize).pow(3) / 128],
                                vec![0u8; (res as usize).pow(3)],
                                None,
                                None,
                                None,
                            )
                        }
                    } else {
                        let (v, c) = voxelize::ply_to_voxels(&path, res, base_palette_index);
                        (v, c, None, Some(res), Some((res, res, res)))
                    };
                let used_res = native_res_opt.unwrap_or(res);
                let (dx, dy, dz) = dims_opt.unwrap_or((used_res, used_res, used_res));
                let _ = txc.send(VoxelJobMessage::Finished {
                    generation: generation_id,
                    index: idx,
                    data: vox,
                    colors,
                    resolution: used_res,
                    dim_x: dx,
                    dim_y: dy,
                    dim_z: dz,
                });
                // transmit palette slice for .vox models so we can blend custom palette
                if let Some(pslice) = palette_opt {
                    let _ = txc.send(VoxelJobMessage::PaletteSlice {
                        generation: generation_id,
                        base: base_palette_index,
                        colors: pslice,
                    });
                }
            });
        }
        drop(tx);
        let voxel_views = vec![None; model_count];
        let color_index_views = vec![None; model_count];
        let voxel_pending = vec![true; model_count];
        let voxel_progress = vec![(0, 0); model_count];
        let voxel_cancel_flag = Arc::new(AtomicBool::new(false));
        let active_voxel_grids = model_count as u32;
        Self {
            models,
            voxel_set,
            voxel_resolution: initial_resolution,
            grid_resolutions,
            grid_dims,
            future_grid_resolutions,
            active_voxel_grids,
            voxel_views,
            color_index_views,
            voxel_pending,
            voxel_progress,
            placeholder_view,
            placeholder_color_view,
            palette_buffer,
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
                VoxelJobMessage::Finished {
                    generation,
                    index,
                    data,
                    colors,
                    resolution,
                    dim_x,
                    dim_y,
                    dim_z,
                } => {
                    if generation != self.voxel_generation || self.cancel_requested {
                        continue;
                    }
                    if index < self.voxel_views.len() {
                        let view = create_voxel_image_view(
                            memory_allocator.clone(),
                            command_buffer_allocator.clone(),
                            queue.clone(),
                            data,
                            resolution,
                        );
                        self.voxel_views[index] = Some(view);
                        // Create color index image view
                        let cview = crate::voxel::create_color_index_image_view(
                            memory_allocator.clone(),
                            command_buffer_allocator.clone(),
                            queue.clone(),
                            colors,
                            resolution,
                        );
                        self.color_index_views[index] = Some(cview);
                        self.voxel_pending[index] = false;
                        if index < self.grid_resolutions.len() {
                            self.grid_resolutions[index] = resolution;
                        }
                        if index < self.grid_dims.len() {
                            self.grid_dims[index] = (dim_x, dim_y, dim_z);
                        }
                        // NOTE: For .vox models, palette slice isn't yet copied into palette buffer.
                        // palette slice (if any) applied separately via PaletteSlice message
                    }
                }
                VoxelJobMessage::PaletteSlice { generation, base, colors } => {
                    if generation != self.voxel_generation || self.cancel_requested {
                        continue;
                    }
                    if !colors.is_empty() {
                        if let Ok(mut data) = self.palette_buffer.write() {
                            let mut dst = base as usize;
                            for c in colors.iter() {
                                if dst >= 256 {
                                    break;
                                }
                                data[dst] = *c;
                                dst += 1;
                            }
                        }
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
            // Rebuild descriptor set including ANY ready grids (relax contiguous requirement).
            let mut ready: Vec<Arc<ImageView>> = Vec::new();
            let mut ready_colors: Vec<Arc<ImageView>> = Vec::new();
            for i in 0..self.voxel_views.len() {
                if let (Some(v), Some(cv)) = (&self.voxel_views[i], &self.color_index_views[i]) {
                    ready.push(v.clone());
                    ready_colors.push(cv.clone());
                }
            }
            self.active_voxel_grids = ready.len() as u32;
            if self.active_voxel_grids > 0 {
                // Build grid info array with precomputed origins.
                // Use actual content width (dim_x) for spacing instead of padded storage resolution.
                let mut infos: Vec<GridInfo> = Vec::new();
                let mut cursor = 0.0f32;
                let mut row_y = 0.0f32; // 2D packing vertical offset accumulator
                for (i, _view) in ready.iter().enumerate() {
                    let res =
                        self.grid_resolutions.get(i).copied().unwrap_or(self.voxel_resolution);
                    let (dx, dy, dz) = self.grid_dims.get(i).copied().unwrap_or((res, res, res));
                    infos.push(GridInfo {
                        resolution: res,
                        origin_x: cursor,
                        origin_y: row_y,
                        dim_x: dx,
                        dim_y: dy,
                        dim_z: dz,
                        _pad0: 0,
                    });
                    // Advance by actual width plus 25% padding of that width
                    cursor += dx as f32 * 1.25;
                    let row_height = dy as f32 * 1.25;
                    // Simple heuristic: wrap when current row width exceeds 1.5 * average width so far or large sentinel
                    // For now use a target aspect: if cursor > (max_y_so_far * 2.0) not available here, approximate by threshold.
                    if cursor > 512.0 {
                        // TODO: dynamic threshold (e.g., sqrt(total area))
                        row_y += row_height;
                        cursor = 0.0;
                    }
                }
                let grid_info_buffer = create_grid_info_buffer(memory_allocator.clone(), &infos);
                self.voxel_set = build_voxel_descriptor_set(
                    descriptor_set_allocator.clone(),
                    render_pipeline,
                    &ready,
                    &ready_colors,
                    self.palette_buffer.clone(),
                    grid_info_buffer,
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
        self.start_background_voxelization(
            descriptor_set_allocator,
            render_pipeline,
            memory_allocator.clone(),
        );
    }

    fn start_background_voxelization(
        &mut self, descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        render_pipeline: &HotReloadComputePipeline, memory_allocator: Arc<StandardMemoryAllocator>,
    ) {
        for v in &mut self.voxel_views {
            *v = None;
        }
        for c in &mut self.color_index_views {
            *c = None;
        }
        self.active_voxel_grids = 0;
        self.voxel_generation = self.voxel_generation.wrapping_add(1);
        self.voxel_pending.fill(true);
        self.cancel_requested = false;
        self.voxel_cancel_flag.store(false, Ordering::Relaxed);
        for p in &mut self.voxel_progress {
            *p = (0, 0);
        }
        let gi = [GridInfo {
            resolution: self.voxel_resolution,
            origin_x: 0.0,
            origin_y: 0.0,
            dim_x: self.voxel_resolution,
            dim_y: self.voxel_resolution,
            dim_z: self.voxel_resolution,
            _pad0: 0,
        }];
        let grid_info_buffer = create_grid_info_buffer(memory_allocator.clone(), &gi);
        self.voxel_set = build_voxel_descriptor_set(
            descriptor_set_allocator.clone(),
            render_pipeline,
            std::slice::from_ref(&self.placeholder_view),
            std::slice::from_ref(&self.placeholder_color_view),
            self.palette_buffer.clone(),
            grid_info_buffer,
        );
        let (tx, rx) = mpsc::channel();
        self.voxel_result_rx = rx;
        let generation_id = self.voxel_generation;
        let cancel_flag = self.voxel_cancel_flag.clone();
        let model_count = self.models.len().max(1);
        for (idx, model) in self.models.iter().enumerate() {
            let txc = tx.clone();
            let path = model.path.clone();
            let gen_thread = generation_id;
            let cancel_local = cancel_flag.clone();
            let res_for_grid = self.grid_resolutions[idx];
            thread::spawn(move || {
                use crate::voxelize::{VoxelProgressCallbacks, ply_to_voxels_with_progress};
                let cancelled = cancel_local.clone();
                let base_palette_index = (idx as u32 * (256 / model_count as u32)) as u8;
                // Matches logic in initial spawn; ensures deterministic palette mapping.
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
                if path.extension().and_then(|e| e.to_str()) == Some("vox") {
                    // No progress callbacks for .vox yet (fast load typically); still send palette slice.
                    let palette_span = (256 / model_count as u32) as u8;
                    if !path.exists() {
                        eprintln!("[voxel] .vox file missing: {}", path.display());
                        let _ = txc.send(VoxelJobMessage::Cancelled { generation: gen_thread });
                    } else if let Some(result) = crate::voxelize_vox::vox_to_voxels(
                        &path,
                        None,
                        base_palette_index,
                        palette_span,
                    ) {
                        let vox = result.voxels;
                        let colors = result.colors;
                        let palette_slice = result.palette;
                        if vox.iter().all(|&u| u == 0) {
                            eprintln!(
                                "[voxel] WARNING: .vox produced empty voxel set: {}",
                                path.display()
                            );
                        }
                        if !cancelled.load(Ordering::Relaxed) {
                            let _ = txc.send(VoxelJobMessage::Finished {
                                generation: gen_thread,
                                index: idx,
                                data: vox,
                                colors,
                                resolution: result.used_resolution,
                                dim_x: result.dim_x,
                                dim_y: result.dim_y,
                                dim_z: result.dim_z,
                            });
                            if !palette_slice.is_empty() {
                                let _ = txc.send(VoxelJobMessage::PaletteSlice {
                                    generation: gen_thread,
                                    base: base_palette_index,
                                    colors: palette_slice,
                                });
                            }
                        }
                    } else {
                        eprintln!("[voxel] Failed to parse .vox file: {}", path.display());
                        let _ = txc.send(VoxelJobMessage::Cancelled { generation: gen_thread });
                    }
                } else {
                    if let Some((vox, colors)) = ply_to_voxels_with_progress(
                        &path,
                        res_for_grid,
                        base_palette_index,
                        prog_cb,
                    ) {
                        if !cancelled.load(Ordering::Relaxed) {
                            let _ = txc.send(VoxelJobMessage::Finished {
                                generation: gen_thread,
                                index: idx,
                                data: vox,
                                colors,
                                resolution: res_for_grid,
                                dim_x: res_for_grid,
                                dim_y: res_for_grid,
                                dim_z: res_for_grid,
                            });
                        } else {
                            let _ = txc.send(VoxelJobMessage::Cancelled { generation: gen_thread });
                        }
                    } else {
                        let _ = txc.send(VoxelJobMessage::Cancelled { generation: gen_thread });
                    }
                }
            });
        }
        drop(tx);
    }
}
