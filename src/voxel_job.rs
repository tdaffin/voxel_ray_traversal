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
        storage_w: u32,
        storage_h: u32,
        storage_d: u32,
    },
    PaletteSlice {
        generation: u64,
        index: usize,
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
    pub grid_storage: Vec<(u32, u32, u32)>, // (storage_w, storage_h, storage_d)
    pub future_grid_resolutions: Vec<u32>,
    pub active_voxel_grids: u32,

    pub voxel_views: Vec<Option<Arc<ImageView>>>,
    pub color_index_views: Vec<Option<Arc<ImageView>>>,
    pub voxel_pending: Vec<bool>,
    pub voxel_progress: Vec<(usize, usize)>,

    placeholder_view: Arc<ImageView>,
    placeholder_color_view: Arc<ImageView>,
    palette_buffer: Subbuffer<[[f32; 4]]>,
    // Step8: collect per-grid palette slices (local indices) for compaction.
    palette_slices: Vec<Option<Vec<[f32; 4]>>>,
    palette_bases: Vec<u32>,
    palette_lens: Vec<u32>,
    voxel_result_rx: Receiver<VoxelJobMessage>,
    voxel_generation: u64,
    cancel_requested: bool,
    voxel_cancel_flag: Arc<AtomicBool>,
    // Octree data
    pub octree_nodes: Vec<Option<Vec<crate::octree::OctNode>>>,
    pub octree_max_depths: Vec<u32>,
    pub octree_node_buffer: Option<Subbuffer<[crate::octree::OctNode]>>,
    pub octree_grid_info_buffer: Option<Subbuffer<[crate::octree::OctreeGridInfo]>>,
    placeholder_octree_node_buffer: Subbuffer<[crate::octree::OctNode]>,
    placeholder_octree_grid_info_buffer: Subbuffer<[crate::octree::OctreeGridInfo]>,
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
        let grid_storage =
            vec![
                (initial_resolution / 4, initial_resolution / 4, initial_resolution / 8);
                model_count
            ];
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
            initial_resolution,
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
        // Placeholder octree buffers (single dummy entries)
        use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
        use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
        let usage = BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST;
        let placeholder_octree_node_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo { usage, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [crate::octree::OctNode::default()].into_iter(),
        )
        .expect("placeholder octree node buffer");
        let placeholder_octree_grid_info_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo { usage, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [crate::octree::OctreeGridInfo::default()].into_iter(),
        )
        .expect("placeholder octree grid info buffer");
        // Initial grid info (single placeholder)
        let gi = [GridInfo {
            resolution: initial_resolution,
            origin_x: 0.0,
            origin_y: 0.0,
            dim_x: initial_resolution,
            dim_y: initial_resolution,
            dim_z: initial_resolution,
            storage_w: initial_resolution / 4,
            storage_h: initial_resolution / 4,
            storage_d: initial_resolution / 8,
            palette_base: 0,
            palette_len: 0,
        }];
        let grid_info_buffer = create_grid_info_buffer(memory_allocator.clone(), &gi);
        let voxel_set = build_voxel_descriptor_set(
            descriptor_set_allocator.clone(),
            render_pipeline,
            std::slice::from_ref(&placeholder_view),
            std::slice::from_ref(&placeholder_color_view),
            palette_buffer.clone(),
            grid_info_buffer.clone(),
            placeholder_octree_node_buffer.clone(),
            placeholder_octree_grid_info_buffer.clone(),
        );
        let (tx, rx) = mpsc::channel();
        let voxel_generation = 1u64;
        for (idx, model) in models.iter().enumerate() {
            let txc = tx.clone();
            let path = model.path.clone();
            let res = grid_resolutions[idx];
            let generation_id = voxel_generation;
            // Legacy base_palette_index removed; indices now local per grid until compaction.
            thread::spawn(move || {
                let palette_span = (256 / model_count as u32) as u8; // soft cap per model prior to compaction
                let (vox, colors, palette_opt, dims_opt, storage_opt) =
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
                        } else if let Some(result) =
                            crate::voxelize_vox::vox_to_voxels(&path, None, palette_span)
                        {
                            let v = result.voxels;
                            let c = result.colors;
                            let p = result.palette;
                            let dims_tuple = (result.dim_x, result.dim_y, result.dim_z);
                            // Currently .vox path still uses cubic packed storage; derive for now
                            let storage_tuple =
                                (result.storage_w, result.storage_h, result.storage_d);
                            if v.iter().all(|&u| u == 0) {
                                eprintln!(
                                    "[voxel] WARNING: .vox produced empty voxel set: {}",
                                    path.display()
                                );
                            }
                            (v, c, Some(p), Some(dims_tuple), Some(storage_tuple))
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
                        // PLY path now uses local 0-based palette indices; pass 0 (ignored in implementation)
                        let (v, c, logical, storage) = voxelize::ply_to_voxels(&path, res);
                        (v, c, None, Some(logical), Some(storage))
                    };
                let used_res = res; // keep legacy resolution for now (could be dim max)
                let (dx, dy, dz) = dims_opt.unwrap_or((used_res, used_res, used_res));
                let (sw, sh, sd) =
                    storage_opt.unwrap_or((used_res / 4, used_res / 4, used_res / 8));
                let _ = txc.send(VoxelJobMessage::Finished {
                    generation: generation_id,
                    index: idx,
                    data: vox,
                    colors,
                    resolution: used_res,
                    dim_x: dx,
                    dim_y: dy,
                    dim_z: dz,
                    storage_w: sw,
                    storage_h: sh,
                    storage_d: sd,
                });
                // transmit palette slice for .vox models so we can blend custom palette
                if let Some(pslice) = palette_opt {
                    let _ = txc.send(VoxelJobMessage::PaletteSlice {
                        generation: generation_id,
                        index: idx,
                        colors: pslice,
                    });
                } else {
                    // PLY placeholder: generate simple 16-color grayscale slice matching previous procedural use
                    let mut slice = Vec::new();
                    for i in 0..16u8 {
                        slice.push([i as f32 / 15.0, i as f32 / 15.0, i as f32 / 15.0, 1.0]);
                    }
                    let _ = txc.send(VoxelJobMessage::PaletteSlice {
                        generation: generation_id,
                        index: idx,
                        colors: slice,
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
            grid_storage,
            future_grid_resolutions,
            active_voxel_grids,
            voxel_views,
            color_index_views,
            voxel_pending,
            voxel_progress,
            placeholder_view,
            placeholder_color_view,
            palette_buffer,
            palette_slices: vec![None; model_count],
            palette_bases: vec![0; model_count],
            palette_lens: vec![0; model_count],
            voxel_result_rx: rx,
            voxel_generation,
            cancel_requested: false,
            voxel_cancel_flag,
            octree_nodes: vec![None; model_count],
            octree_max_depths: vec![0; model_count],
            octree_node_buffer: None,
            octree_grid_info_buffer: None,
            placeholder_octree_node_buffer,
            placeholder_octree_grid_info_buffer,
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
                    storage_w,
                    storage_h,
                    storage_d,
                } => {
                    if generation != self.voxel_generation || self.cancel_requested {
                        continue;
                    }
                    if index < self.voxel_views.len() {
                        let voxel_data_clone = data.clone(); // clone for octree build
                        let view = create_voxel_image_view(
                            memory_allocator.clone(),
                            command_buffer_allocator.clone(),
                            queue.clone(),
                            data,
                            storage_w,
                            storage_h,
                            storage_d,
                        );
                        self.voxel_views[index] = Some(view);
                        // Create color index image view
                        let cview = crate::voxel::create_color_index_image_view(
                            memory_allocator.clone(),
                            command_buffer_allocator.clone(),
                            queue.clone(),
                            colors,
                            dim_x,
                            dim_y,
                            dim_z,
                        );
                        self.color_index_views[index] = Some(cview);
                        self.voxel_pending[index] = false;
                        if index < self.grid_resolutions.len() {
                            self.grid_resolutions[index] = resolution; // keep logical base resolution for now
                        }
                        if index < self.grid_dims.len() {
                            self.grid_dims[index] = (dim_x, dim_y, dim_z);
                        }
                        if index < self.grid_storage.len() {
                            self.grid_storage[index] = (storage_w, storage_h, storage_d);
                        }
                        // Build octree (synchronously for now)
                        let (dimx, dimy, dimz) = (dim_x, dim_y, dim_z);
                        let blocks_w = storage_w; // number of 4x4x8 blocks in x
                        let blocks_h = storage_h;
                        let blocks_d = storage_d;
                        let get_occ = |x: u32, y: u32, z: u32| -> bool {
                            if x >= dimx || y >= dimy || z >= dimz {
                                return false;
                            }
                            let block_x = x / 4;
                            let block_y = y / 4;
                            let block_z = z / 8;
                            if block_x >= blocks_w || block_y >= blocks_h || block_z >= blocks_d {
                                return false;
                            }
                            let index_b = (block_z * blocks_h * blocks_w
                                + block_y * blocks_w
                                + block_x) as usize;
                            if index_b >= voxel_data_clone.len() {
                                return false;
                            }
                            let texel = voxel_data_clone[index_b];
                            let lane = x % 4; // selects u32 within u128
                            let word = ((texel >> (lane * 32)) & 0xFFFF_FFFF) as u32;
                            let bit_index = (y % 4) + (z % 8) * 4;
                            ((word >> bit_index) & 1) != 0
                        };
                        let (nodes, max_depth) =
                            crate::octree::build_octree((dimx, dimy, dimz), get_occ, Some(6));
                        if index < self.octree_nodes.len() {
                            self.octree_nodes[index] = Some(nodes);
                        }
                        if index < self.octree_max_depths.len() {
                            self.octree_max_depths[index] = max_depth;
                        }
                        // NOTE: For .vox models, palette slice isn't yet copied into palette buffer.
                        // palette slice (if any) applied separately via PaletteSlice message
                        // TODO Step8: store per-grid palette_base/len in parallel vectors, update later
                    }
                }
                VoxelJobMessage::PaletteSlice { generation, index, colors } => {
                    if generation != self.voxel_generation || self.cancel_requested {
                        continue;
                    }
                    if index < self.palette_slices.len() {
                        self.palette_slices[index] = Some(colors);
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
                // Refined heuristic:
                // 1. Compute approximate total area (dx*dy) and target row width ~= sqrt(total_area) * k
                // 2. Greedy pack grids into rows until adding next would exceed target, then wrap.
                // 3. Within a row, lay out left-to-right with padding factor P.
                const PADDING: f32 = 0.10; // 10% spacing (reduced from 25%)
                let mut dims_ready: Vec<(usize, u32, u32, u32, u32)> = Vec::new();
                let mut total_area: f64 = 0.0;
                for (i, _view) in ready.iter().enumerate() {
                    let res =
                        self.grid_resolutions.get(i).copied().unwrap_or(self.voxel_resolution);
                    let (dx, dy, dz) = self.grid_dims.get(i).copied().unwrap_or((res, res, res));
                    total_area += (dx.max(1) * dy.max(1)) as f64;
                    dims_ready.push((i, res, dx, dy, dz));
                }
                // Sort by descending height to improve packing (tallest-first skyline approximation)
                dims_ready.sort_by_key(|&(_, _, _dx, dy, _)| std::cmp::Reverse(dy));
                let target_row_width = (total_area.sqrt() as f32).max(1.0);
                let mut infos: Vec<GridInfo> = vec![GridInfo::default(); dims_ready.len()];
                let mut row_y = 0.0f32;
                let mut cursor = 0.0f32; // projected row width accumulator
                let mut current_row: Vec<(usize, u32, u32, u32, u32)> = Vec::new();
                let place_row = |row: &Vec<(usize, u32, u32, u32, u32)>,
                                 base_y: f32,
                                 infos: &mut [GridInfo]| {
                    let mut x = 0.0f32;
                    let mut max_h = 0.0f32;
                    for &(orig_index, res, dx, dy, dz) in row.iter() {
                        let (storage_w, storage_h, storage_d) = self
                            .grid_storage
                            .get(orig_index)
                            .copied()
                            .unwrap_or((res / 4, res / 4, res / 8));
                        infos[orig_index] = GridInfo {
                            resolution: res,
                            origin_x: x,
                            origin_y: base_y,
                            dim_x: dx,
                            dim_y: dy,
                            dim_z: dz,
                            storage_w,
                            storage_h,
                            storage_d,
                            palette_base: 0,
                            palette_len: 256,
                        };
                        x += dx as f32 * (1.0 + PADDING);
                        max_h = max_h.max(dy as f32 * (1.0 + PADDING));
                    }
                    max_h
                };
                for entry in dims_ready.into_iter() {
                    let (_, _res, dx, _dy, _dz) = entry;
                    let projected = if current_row.is_empty() {
                        dx as f32
                    } else {
                        cursor + dx as f32 * (1.0 + PADDING)
                    };
                    if !current_row.is_empty() && projected > target_row_width * 1.25 {
                        // allow some slack
                        // flush row
                        let used_h = place_row(&current_row, row_y, &mut infos);
                        row_y += used_h;
                        current_row.clear();
                        // cursor reset not needed; will be set below for first element of new row
                    }
                    cursor = if current_row.is_empty() {
                        dx as f32 * (1.0 + PADDING)
                    } else {
                        projected
                    };
                    current_row.push(entry);
                }
                if !current_row.is_empty() {
                    let _ = place_row(&current_row, row_y, &mut infos);
                }
                // If all grids are ready and we have palette slices for each, perform compaction.
                let all_ready = self.voxel_pending.iter().all(|p| !*p)
                    && self.palette_slices.iter().all(|s| s.is_some());
                if all_ready {
                    // Build compact palette by concatenation; assign bases.
                    let mut compact: Vec<[f32; 4]> = Vec::new();
                    for (i, slice_opt) in self.palette_slices.iter().enumerate() {
                        if let Some(slice) = slice_opt {
                            self.palette_bases[i] = compact.len() as u32;
                            self.palette_lens[i] = slice.len() as u32;
                            compact.extend_from_slice(slice);
                        } else {
                            self.palette_bases[i] = 0;
                            self.palette_lens[i] = 0;
                        }
                    }
                    if compact.is_empty() {
                        compact.push([1.0, 1.0, 1.0, 1.0]);
                    }
                    // Recreate palette buffer with compact data
                    self.palette_buffer =
                        crate::voxel::create_palette_buffer(memory_allocator.clone(), &compact);
                    // Update infos with palette base/len
                    for (i, gi) in infos.iter_mut().enumerate() {
                        gi.palette_base = self.palette_bases.get(i).copied().unwrap_or(0);
                        gi.palette_len = self.palette_lens.get(i).copied().unwrap_or(0);
                    }
                }
                let grid_info_buffer = create_grid_info_buffer(memory_allocator.clone(), &infos);
                // Build or update octree buffers if all_ready and octrees exist for grids; concatenate per-grid nodes.
                if all_ready {
                    let mut concatenated: Vec<crate::octree::OctNode> = Vec::new();
                    let mut oct_infos: Vec<crate::octree::OctreeGridInfo> = Vec::new();
                    for (i, _gi) in infos.iter().enumerate() {
                        if let Some(onodes) = self.octree_nodes.get(i).and_then(|o| o.as_ref()) {
                            let offset = concatenated.len() as u32;
                            concatenated.extend_from_slice(onodes);
                            oct_infos.push(crate::octree::OctreeGridInfo {
                                node_offset: offset,
                                node_count: onodes.len() as u32,
                                max_depth: self.octree_max_depths.get(i).copied().unwrap_or(0),
                                flags: 1,
                            });
                        } else {
                            oct_infos.push(crate::octree::OctreeGridInfo {
                                node_offset: 0,
                                node_count: 0,
                                max_depth: 0,
                                flags: 0,
                            });
                        }
                    }
                    if concatenated.is_empty() {
                        concatenated.push(Default::default());
                    }
                    if oct_infos.is_empty() {
                        oct_infos.push(Default::default());
                    }
                    use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
                    use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
                    let usage = BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST;
                    self.octree_node_buffer = Some(
                        Buffer::from_iter(
                            memory_allocator.clone(),
                            BufferCreateInfo { usage, ..Default::default() },
                            AllocationCreateInfo {
                                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                                ..Default::default()
                            },
                            concatenated.into_iter(),
                        )
                        .expect("octree node buffer"),
                    );
                    self.octree_grid_info_buffer = Some(
                        Buffer::from_iter(
                            memory_allocator.clone(),
                            BufferCreateInfo { usage, ..Default::default() },
                            AllocationCreateInfo {
                                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                                ..Default::default()
                            },
                            oct_infos.into_iter(),
                        )
                        .expect("octree grid info buffer"),
                    );
                }
                self.voxel_set = build_voxel_descriptor_set(
                    descriptor_set_allocator.clone(),
                    render_pipeline,
                    &ready,
                    &ready_colors,
                    self.palette_buffer.clone(),
                    grid_info_buffer,
                    self.octree_node_buffer
                        .clone()
                        .unwrap_or(self.placeholder_octree_node_buffer.clone()),
                    self.octree_grid_info_buffer
                        .clone()
                        .unwrap_or(self.placeholder_octree_grid_info_buffer.clone()),
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
                // Do not override native .vox model resolutions; they come from the file.
                let is_vox =
                    self.models.get(i).map(|m| m.extension.as_str() == "vox").unwrap_or(false);
                if !is_vox {
                    self.grid_resolutions[i] = r.div_ceil(8) * 8;
                }
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
            storage_w: self.voxel_resolution / 4,
            storage_h: self.voxel_resolution / 4,
            storage_d: self.voxel_resolution / 8,
            palette_base: 0,
            palette_len: 0,
        }];
        let grid_info_buffer = create_grid_info_buffer(memory_allocator.clone(), &gi);
        self.voxel_set = build_voxel_descriptor_set(
            descriptor_set_allocator.clone(),
            render_pipeline,
            std::slice::from_ref(&self.placeholder_view),
            std::slice::from_ref(&self.placeholder_color_view),
            self.palette_buffer.clone(),
            grid_info_buffer,
            self.placeholder_octree_node_buffer.clone(),
            self.placeholder_octree_grid_info_buffer.clone(),
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
                // Base palette index legacy removed; indices are local until compaction.
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
                    } else if let Some(result) =
                        crate::voxelize_vox::vox_to_voxels(&path, None, palette_span)
                    {
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
                                storage_w: result.storage_w,
                                storage_h: result.storage_h,
                                storage_d: result.storage_d,
                            });
                            if !palette_slice.is_empty() {
                                let _ = txc.send(VoxelJobMessage::PaletteSlice {
                                    generation: gen_thread,
                                    index: idx,
                                    colors: palette_slice,
                                });
                            }
                        }
                    } else {
                        eprintln!("[voxel] Failed to parse .vox file: {}", path.display());
                        let _ = txc.send(VoxelJobMessage::Cancelled { generation: gen_thread });
                    }
                } else {
                    if let Some((vox, colors, _logical, _storage)) =
                        ply_to_voxels_with_progress(&path, res_for_grid, prog_cb)
                    {
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
                                storage_w: res_for_grid / 4,
                                storage_h: res_for_grid / 4,
                                storage_d: res_for_grid / 8,
                            });
                            // Emit procedural palette slice (grayscale ramp)
                            let mut slice = Vec::new();
                            for i in 0..16u8 {
                                slice.push([
                                    i as f32 / 15.0,
                                    i as f32 / 15.0,
                                    i as f32 / 15.0,
                                    1.0,
                                ]);
                            }
                            let _ = txc.send(VoxelJobMessage::PaletteSlice {
                                generation: gen_thread,
                                index: idx,
                                colors: slice,
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
