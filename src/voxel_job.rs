use std::f32::consts::TAU;
use std::sync::mpsc::{self, Receiver};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::{DescriptorSet, allocator::StandardDescriptorSetAllocator};
use vulkano::device::Queue;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::Pipeline;

use crate::{
    hot_reload::HotReloadComputePipeline,
    model_discovery::{DiscoveredModel, discover_models},
    voxel::{
        GridInfo, TilePayloadGpu, build_voxel_descriptor_set, create_empty_voxel_placeholder,
        create_grid_info_buffer, create_tile_mask_image_view, create_tile_payload_buffer,
        create_voxel_image_view,
    },
    voxelize,
};

fn identity_mat4() -> [[f32; 4]; 4] {
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

fn compose_transform(rotation: [[f32; 4]; 4], translation: [f32; 3]) -> [[f32; 4]; 4] {
    let mut mat = rotation;
    mat[3][0] = translation[0];
    mat[3][1] = translation[1];
    mat[3][2] = translation[2];
    mat
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq <= f32::EPSILON {
        [0.0, 0.0, 1.0]
    } else {
        let inv_len = len_sq.sqrt().recip();
        [v[0] * inv_len, v[1] * inv_len, v[2] * inv_len]
    }
}

fn rotation_from_axis_angle(axis: [f32; 3], angle: f32) -> [[f32; 4]; 4] {
    let n = normalize3(axis);
    let (x, y, z) = (n[0], n[1], n[2]);
    let cos = angle.cos();
    let sin = angle.sin();
    let one_minus = 1.0 - cos;
    [
        [one_minus * x * x + cos, one_minus * x * y - sin * z, one_minus * x * z + sin * y, 0.0],
        [one_minus * x * y + sin * z, one_minus * y * y + cos, one_minus * y * z - sin * x, 0.0],
        [one_minus * x * z - sin * y, one_minus * y * z + sin * x, one_minus * z * z + cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] =
                a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

fn random_rotation_matrix<R: Rng + ?Sized>(rng: &mut R) -> [[f32; 4]; 4] {
    // Marsaglia method for random unit vector
    let u: f32 = rng.gen_range(-1.0..=1.0);
    let theta: f32 = rng.gen_range(0.0..TAU);
    let sqrt_term = (1.0 - u * u).max(0.0).sqrt();
    let axis = [sqrt_term * theta.cos(), sqrt_term * theta.sin(), u];
    let angle: f32 = rng.gen_range(0.0..TAU);
    rotation_from_axis_angle(axis, angle)
}

fn translation_to_mat4(translation: [f32; 3]) -> [[f32; 4]; 4] {
    compose_transform(identity_mat4(), translation)
}

fn scale_translation_mat4(scale: [f32; 3], translation: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [scale[0], 0.0, 0.0, 0.0],
        [0.0, scale[1], 0.0, 0.0],
        [0.0, 0.0, scale[2], 0.0],
        [translation[0], translation[1], translation[2], 1.0],
    ]
}

fn seeded_rng_for_path(path: &std::path::Path) -> ChaCha8Rng {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    let seed = hasher.finish();
    let mut seed_bytes = [0u8; 32];
    seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
    ChaCha8Rng::from_seed(seed_bytes)
}

#[derive(Debug)]
pub enum VoxelJobMessage {
    Finished {
        generation: u64,
        index: usize,
        colors: Vec<u8>,
        resolution: u32,
        dim_x: u32,
        dim_y: u32,
        dim_z: u32,
        storage_w: u32,
        storage_h: u32,
        storage_d: u32,
        tile_mask: Vec<u32>,
        tile_payloads: Vec<[u32; 4]>,
        tile_stats: crate::tile_compression::TileCompressionStats,
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
    pub grid_user_transforms: Vec<[[f32; 4]; 4]>,
    pub grid_rotation_speeds: Vec<f32>, // radians per second
    pub future_grid_resolutions: Vec<u32>,
    pub active_voxel_grids: u32,
    voxel_images_enabled: bool,

    pub voxel_views: Vec<Option<Arc<ImageView>>>,
    pub color_index_views: Vec<Option<Arc<ImageView>>>,
    pub voxel_pending: Vec<bool>,
    pub voxel_progress: Vec<(usize, usize)>,

    pub tile_mask_views: Vec<Option<Arc<ImageView>>>,
    pub tile_payloads: Vec<Option<Vec<[u32; 4]>>>,
    pub tile_stats: Vec<Option<crate::tile_compression::TileCompressionStats>>,

    placeholder_view: Arc<ImageView>,
    placeholder_color_view: Arc<ImageView>,
    placeholder_tile_mask_view: Arc<ImageView>,
    ground_tile_mask_view: Arc<ImageView>,
    palette_buffer: Subbuffer<[[f32; 4]]>,
    tile_payload_buffer: Subbuffer<[TilePayloadGpu]>,
    // Step8: collect per-grid palette slices (local indices) for compaction.
    palette_slices: Vec<Option<Vec<[f32; 4]>>>,
    palette_bases: Vec<u32>,
    palette_lens: Vec<u32>,
    ground_voxel_view: Arc<ImageView>,
    ground_color_view: Arc<ImageView>,
    ground_resolution: u32,
    ground_dims: (u32, u32, u32),
    ground_storage: (u32, u32, u32),
    ground_palette_slice: Vec<[f32; 4]>,
    ground_palette_base: u32,
    ground_palette_len: u32,
    grid_rotation_angles: Vec<f32>,
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
        let voxel_images_enabled = render_pipeline.layout().set_layouts()[1]
            .bindings()
            .contains_key(&crate::voxel::BINDING_VOXEL_IMAGES);
        let grid_resolutions = vec![initial_resolution; model_count];
        let grid_dims =
            vec![(initial_resolution, initial_resolution, initial_resolution); model_count];
        let grid_storage =
            vec![
                (initial_resolution / 4, initial_resolution / 4, initial_resolution / 8);
                model_count
            ];
        let grid_user_transforms = vec![identity_mat4(); model_count];
        let grid_rotation_speeds = vec![0.0f32; model_count];
        let grid_rotation_angles = vec![0.0f32; model_count];
        let future_grid_resolutions = grid_resolutions.clone();
        let placeholder_view = create_empty_voxel_placeholder(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            if voxel_images_enabled { initial_resolution } else { 8 },
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
        let ground_palette_slice = vec![[0.2, 0.55, 0.25, 1.0]]; // grassy green
        let ground_palette_base = palette.len() as u32;
        let ground_palette_len = ground_palette_slice.len() as u32;
        palette.extend_from_slice(&ground_palette_slice);
        let palette_buffer =
            crate::voxel::create_palette_buffer(memory_allocator.clone(), &palette);
        let ground_resolution = 1u32;
        let ground_dims = (1u32, 1u32, 1u32);
        let ground_storage = (1u32, 1u32, 1u32);
        let ground_voxel_view = if voxel_images_enabled {
            let ground_voxels = vec![1u128];
            create_voxel_image_view(
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                queue.clone(),
                ground_voxels,
                ground_storage.0,
                ground_storage.1,
                ground_storage.2,
            )
        } else {
            placeholder_view.clone()
        };
        let ground_colors = vec![0u8; (ground_dims.0 * ground_dims.1 * ground_dims.2) as usize];
        let ground_color_view = crate::voxel::create_color_index_image_view(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            ground_colors,
            ground_dims.0,
            ground_dims.1,
            ground_dims.2,
        );
        let storage_w_placeholder = (initial_resolution / 4).max(1);
        let storage_h_placeholder = (initial_resolution / 4).max(1);
        let storage_d_placeholder = (initial_resolution / 8).max(1);
        let mask_texel_count =
            (storage_w_placeholder * storage_h_placeholder * storage_d_placeholder).max(1);
        let placeholder_mask_entries = vec![0u32; mask_texel_count as usize];
        let placeholder_tile_mask_view = create_tile_mask_image_view(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            placeholder_mask_entries,
            storage_w_placeholder,
            storage_h_placeholder,
            storage_d_placeholder,
        );
        let ground_mask_entries = vec![
            crate::tile_compression::TILE_UNIFORM_BIT
                | crate::tile_compression::TILE_UNIFORM_VALUE_BIT,
        ];
        let ground_tile_mask_view = create_tile_mask_image_view(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            ground_mask_entries,
            ground_storage.0,
            ground_storage.1,
            ground_storage.2,
        );
        let placeholder_payload_buffer =
            create_tile_payload_buffer(memory_allocator.clone(), &[TilePayloadGpu::default()]);
        // Initial grid info (single placeholder)
        let gi = [GridInfo {
            resolution: initial_resolution,
            dim_x: initial_resolution,
            dim_y: initial_resolution,
            dim_z: initial_resolution,
            storage_w: initial_resolution / 4,
            storage_h: initial_resolution / 4,
            storage_d: initial_resolution / 8,
            palette_base: 0,
            palette_len: 0,
            grid_to_world: translation_to_mat4([0.0, 0.0, 0.0]),
            ..GridInfo::default()
        }];
        let grid_info_buffer = create_grid_info_buffer(memory_allocator.clone(), &gi);
        let voxel_set = build_voxel_descriptor_set(
            descriptor_set_allocator.clone(),
            render_pipeline,
            std::slice::from_ref(&placeholder_view),
            std::slice::from_ref(&placeholder_color_view),
            std::slice::from_ref(&placeholder_tile_mask_view),
            palette_buffer.clone(),
            placeholder_payload_buffer.clone(),
            grid_info_buffer.clone(),
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
                let palette_span: u8 = u8::MAX; // allow .vox models to retain all colors
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
                        let (v, _c, logical, storage) = voxelize::ply_to_voxels(&path, res);
                        let (dx, dy, dz) = logical;
                        let color_count = dx as usize * dy as usize * dz as usize;
                        let colors = vec![0u8; color_count];
                        let mut rng = seeded_rng_for_path(&path);
                        let palette_color = [
                            rng.gen_range(0.0..1.0),
                            rng.gen_range(0.0..1.0),
                            rng.gen_range(0.0..1.0),
                            1.0,
                        ];
                        (v, colors, Some(vec![palette_color]), Some(logical), Some(storage))
                    };
                let used_res = res; // keep legacy resolution for now (could be dim max)
                let (dx, dy, dz) = dims_opt.unwrap_or((used_res, used_res, used_res));
                let (sw, sh, sd) =
                    storage_opt.unwrap_or((used_res / 4, used_res / 4, used_res / 8));
                let compression = if dx == 0 || dy == 0 || dz == 0 {
                    crate::tile_compression::TileCompressionResult::default()
                } else {
                    crate::tile_compression::classify_tiles(&vox, (dx, dy, dz), (sw, sh, sd))
                };
                let tile_mask = compression.mask_entries;
                let tile_payloads: Vec<[u32; 4]> =
                    compression.payloads.into_iter().map(|p| p.occupancy).collect();
                let tile_stats = compression.stats;
                let _ = txc.send(VoxelJobMessage::Finished {
                    generation: generation_id,
                    index: idx,
                    colors,
                    resolution: used_res,
                    dim_x: dx,
                    dim_y: dy,
                    dim_z: dz,
                    storage_w: sw,
                    storage_h: sh,
                    storage_d: sd,
                    tile_mask,
                    tile_payloads,
                    tile_stats,
                });
                // transmit palette slice for .vox models so we can blend custom palette
                if let Some(pslice) = palette_opt {
                    let _ = txc.send(VoxelJobMessage::PaletteSlice {
                        generation: generation_id,
                        index: idx,
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
            grid_storage,
            grid_user_transforms,
            grid_rotation_speeds,
            future_grid_resolutions,
            active_voxel_grids,
            voxel_images_enabled,
            voxel_views,
            color_index_views,
            voxel_pending,
            voxel_progress,
            tile_mask_views: vec![None; model_count],
            tile_payloads: vec![None; model_count],
            tile_stats: vec![None; model_count],
            placeholder_view,
            placeholder_color_view,
            placeholder_tile_mask_view,
            ground_tile_mask_view,
            palette_buffer,
            tile_payload_buffer: placeholder_payload_buffer,
            palette_slices: vec![None; model_count],
            palette_bases: vec![0; model_count],
            palette_lens: vec![0; model_count],
            ground_voxel_view,
            ground_color_view,
            ground_resolution,
            ground_dims,
            ground_storage,
            ground_palette_slice,
            ground_palette_base,
            ground_palette_len,
            grid_rotation_angles,
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
        delta_seconds: f32,
    ) {
        let mut dirty = false;
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
                    colors,
                    resolution,
                    dim_x,
                    dim_y,
                    dim_z,
                    storage_w,
                    storage_h,
                    storage_d,
                    tile_mask,
                    tile_payloads,
                    tile_stats,
                } => {
                    if crate::log_config::verbose_logging() {
                        eprintln!(
                            "[voxel] grid {index} compression: empty={} uniform={} dense={} (payloads={})",
                            tile_stats.empty_tiles,
                            tile_stats.uniform_tiles,
                            tile_stats.dense_tiles,
                            tile_payloads.len()
                        );
                    }
                    if generation != self.voxel_generation || self.cancel_requested {
                        continue;
                    }
                    if index < self.voxel_views.len() {
                        assert!(
                            !self.voxel_images_enabled,
                            "voxel image binding present but packed data uploads were removed"
                        );
                        self.voxel_views[index] = Some(self.placeholder_view.clone());
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
                        if index < self.tile_mask_views.len() {
                            let mask_view = create_tile_mask_image_view(
                                memory_allocator.clone(),
                                command_buffer_allocator.clone(),
                                queue.clone(),
                                tile_mask,
                                storage_w,
                                storage_h,
                                storage_d,
                            );
                            self.tile_mask_views[index] = Some(mask_view);
                        }
                        if index < self.tile_payloads.len() {
                            self.tile_payloads[index] = Some(tile_payloads);
                        }
                        if index < self.tile_stats.len() {
                            self.tile_stats[index] = Some(tile_stats);
                        }
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
                        // NOTE: For .vox models, palette slice isn't yet copied into palette buffer.
                        // palette slice (if any) applied separately via PaletteSlice message
                        // TODO Step8: store per-grid palette_base/len in parallel vectors, update later
                        dirty = true;
                    }
                }
                VoxelJobMessage::PaletteSlice { generation, index, colors } => {
                    if generation != self.voxel_generation || self.cancel_requested {
                        continue;
                    }
                    if index < self.palette_slices.len() {
                        self.palette_slices[index] = Some(colors);
                        dirty = true;
                    }
                }
                VoxelJobMessage::Cancelled { generation } => {
                    if generation != self.voxel_generation {
                        continue;
                    }
                    if self.cancel_requested {
                        self.voxel_pending.fill(false);
                        dirty = true;
                    }
                }
            }
        }
        let rotations_dirty = self.advance_rotation_angles(delta_seconds);
        if dirty || rotations_dirty {
            self.rebuild_descriptor_set(
                descriptor_set_allocator,
                render_pipeline,
                memory_allocator,
            );
        }
    }

    fn ensure_transform_capacity(&mut self) {
        if self.grid_user_transforms.len() < self.models.len() {
            let needed = self.models.len() - self.grid_user_transforms.len();
            self.grid_user_transforms.extend((0..needed).map(|_| identity_mat4()));
            self.grid_rotation_speeds.extend(std::iter::repeat(0.0).take(needed));
            self.grid_rotation_angles.extend(std::iter::repeat(0.0).take(needed));
        }
    }

    fn advance_rotation_angles(&mut self, delta_seconds: f32) -> bool {
        if delta_seconds <= 0.0 {
            return false;
        }
        self.ensure_transform_capacity();
        let mut changed = false;
        for (angle, speed) in
            self.grid_rotation_angles.iter_mut().zip(self.grid_rotation_speeds.iter())
        {
            if speed.abs() > f32::EPSILON {
                *angle = (*angle + speed * delta_seconds).rem_euclid(TAU);
                changed = true;
            }
        }
        changed
    }

    fn place_row(
        &self, row: &[(usize, usize, u32, u32, u32, u32)], base_y: f32, padding: f32,
        infos: &mut [GridInfo], layout_extents: &mut Vec<(f32, f32, f32, f32)>,
    ) -> f32 {
        let mut x = 0.0f32;
        let mut max_h = 0.0f32;
        for &(descriptor_idx, orig_index, res, dx, dy, dz) in row.iter() {
            let (storage_w, storage_h, storage_d) =
                self.grid_storage.get(orig_index).copied().unwrap_or((res / 4, res / 4, res / 8));
            let mut rotation =
                self.grid_user_transforms.get(orig_index).cloned().unwrap_or_else(identity_mat4);
            if let Some(angle) = self.grid_rotation_angles.get(orig_index) {
                if angle.abs() > f32::EPSILON {
                    let yaw = rotation_from_axis_angle([0.0, 0.0, 1.0], *angle);
                    rotation = mat4_mul(rotation, yaw);
                }
            }
            let center = [dx as f32 * 0.5, dy as f32 * 0.5, dz as f32 * 0.5];
            let to_center = translation_to_mat4([-center[0], -center[1], -center[2]]);
            let back_center = translation_to_mat4(center);
            let base_translation = translation_to_mat4([x, base_y, 0.0]);
            let transform =
                mat4_mul(mat4_mul(mat4_mul(to_center, rotation), back_center), base_translation);
            infos[descriptor_idx] = GridInfo {
                resolution: res,
                dim_x: dx,
                dim_y: dy,
                dim_z: dz,
                storage_w,
                storage_h,
                storage_d,
                palette_base: 0,
                palette_len: 0,
                grid_to_world: transform,
                ..GridInfo::default()
            };
            let center_x = x + dx as f32 * 0.5;
            let center_y = base_y + dy as f32 * 0.5;
            let radius = 0.5 * ((dx as f32).hypot(dy as f32));
            let base_z = infos[descriptor_idx].grid_to_world[3][2];
            layout_extents.push((center_x, center_y, radius, base_z));
            x += dx as f32 * (1.0 + padding);
            max_h = max_h.max(dy as f32 * (1.0 + padding));
        }
        max_h
    }

    fn rebuild_descriptor_set(
        &mut self, descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        render_pipeline: &HotReloadComputePipeline, memory_allocator: Arc<StandardMemoryAllocator>,
    ) {
        let mut ready: Vec<Arc<ImageView>> = Vec::new();
        let mut ready_colors: Vec<Arc<ImageView>> = Vec::new();
        let mut ready_masks: Vec<Arc<ImageView>> = Vec::new();
        let mut ready_indices: Vec<usize> = Vec::new();
        for (idx, (voxel_opt, color_opt)) in
            self.voxel_views.iter().zip(self.color_index_views.iter()).enumerate()
        {
            if let (Some(v), Some(cv)) = (voxel_opt, color_opt) {
                ready.push(v.clone());
                ready_colors.push(cv.clone());
                let mask_view = self
                    .tile_mask_views
                    .get(idx)
                    .and_then(|opt| opt.clone())
                    .unwrap_or_else(|| self.placeholder_tile_mask_view.clone());
                ready_masks.push(mask_view);
                ready_indices.push(idx);
            }
        }

        if ready.is_empty() {
            self.active_voxel_grids = 0;
            return;
        }

        self.ensure_transform_capacity();

        const PADDING: f32 = 0.10;
        let mut dims_ready: Vec<(usize, usize, u32, u32, u32, u32)> = Vec::new();
        let mut total_area: f64 = 0.0;
        for (descriptor_idx, &orig_index) in ready_indices.iter().enumerate() {
            let res =
                self.grid_resolutions.get(orig_index).copied().unwrap_or(self.voxel_resolution);
            let (dx, dy, dz) = self.grid_dims.get(orig_index).copied().unwrap_or((res, res, res));
            total_area += (dx.max(1) * dy.max(1)) as f64;
            dims_ready.push((descriptor_idx, orig_index, res, dx, dy, dz));
        }

        dims_ready.sort_by_key(|&(_, _, _, _, dy, _)| std::cmp::Reverse(dy));

        let target_row_width = (total_area.sqrt() as f32).max(1.0);
        let mut infos: Vec<GridInfo> = vec![GridInfo::default(); ready.len()];
        let mut layout_extents: Vec<(f32, f32, f32, f32)> = Vec::with_capacity(ready.len());
        let mut row_y = 0.0f32;
        let mut cursor = 0.0f32;
        let mut current_row: Vec<(usize, usize, u32, u32, u32, u32)> = Vec::new();

        for entry in dims_ready.into_iter() {
            let (_, _, _, dx, _, _) = entry;
            let projected = if current_row.is_empty() {
                dx as f32
            } else {
                cursor + dx as f32 * (1.0 + PADDING)
            };
            if !current_row.is_empty() && projected > target_row_width * 1.25 {
                let used_h =
                    self.place_row(&current_row, row_y, PADDING, &mut infos, &mut layout_extents);
                row_y += used_h;
                current_row.clear();
            }
            cursor = if current_row.is_empty() { dx as f32 * (1.0 + PADDING) } else { projected };
            current_row.push(entry);
        }
        if !current_row.is_empty() {
            let _ = self.place_row(&current_row, row_y, PADDING, &mut infos, &mut layout_extents);
        }

        let mut ground_transform_opt: Option<[[f32; 4]; 4]> = None;
        if !layout_extents.is_empty() {
            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;
            let mut min_y = f32::MAX;
            let mut max_y = f32::MIN;
            let mut min_z = f32::MAX;
            for (cx, cy, radius, base_z) in layout_extents.iter() {
                min_x = min_x.min(cx - radius);
                max_x = max_x.max(cx + radius);
                min_y = min_y.min(cy - radius);
                max_y = max_y.max(cy + radius);
                min_z = min_z.min(*base_z);
            }
            let width = (max_x - min_x).max(1.0);
            let depth = (max_y - min_y).max(1.0);
            let margin_ratio = 0.10;
            let width_margin = width * margin_ratio;
            let depth_margin = depth * margin_ratio;
            let scale_x = width + width_margin;
            let scale_y = depth + depth_margin;
            let scale_z = 1.0;
            let origin_x = min_x - width_margin * 0.5;
            let origin_y = min_y - depth_margin * 0.5;
            let origin_z = min_z - scale_z;
            let transform =
                scale_translation_mat4([scale_x, scale_y, scale_z], [origin_x, origin_y, origin_z]);
            ground_transform_opt = Some(transform);
        }

        let mut compact: Vec<[f32; 4]> = Vec::new();
        for (i, slice_opt) in self.palette_slices.iter().enumerate() {
            let base = compact.len() as u32;
            match slice_opt {
                Some(slice) if !slice.is_empty() => {
                    self.palette_bases[i] = base;
                    self.palette_lens[i] = slice.len() as u32;
                    compact.extend_from_slice(slice);
                }
                _ => {
                    self.palette_bases[i] = base;
                    self.palette_lens[i] = 1;
                    compact.push([1.0, 1.0, 1.0, 1.0]); // placeholder until palette slice arrives
                }
            }
        }
        let ground_base = compact.len() as u32;
        if self.ground_palette_slice.is_empty() {
            compact.push([0.2, 0.55, 0.25, 1.0]);
            self.ground_palette_len = 1;
        } else {
            compact.extend_from_slice(&self.ground_palette_slice);
            self.ground_palette_len = self.ground_palette_slice.len() as u32;
        }
        if compact.is_empty() {
            compact.push([1.0, 1.0, 1.0, 1.0]);
        }
        self.ground_palette_base = ground_base;
        self.palette_buffer =
            crate::voxel::create_palette_buffer(memory_allocator.clone(), &compact);

        let mut payloads_gpu: Vec<TilePayloadGpu> = Vec::new();
        for (descriptor_idx, &orig_index) in ready_indices.iter().enumerate() {
            let base = payloads_gpu.len() as u32;
            infos[descriptor_idx].tile_payload_base = base;
            if let Some(payloads) = self.tile_payloads.get(orig_index).and_then(|opt| opt.as_ref())
            {
                infos[descriptor_idx].tile_payload_len = payloads.len() as u32;
                payloads_gpu
                    .extend(payloads.iter().map(|words| TilePayloadGpu { occupancy: *words }));
            } else {
                infos[descriptor_idx].tile_payload_len = 0;
            }
            infos[descriptor_idx].palette_base =
                self.palette_bases.get(orig_index).copied().unwrap_or(0);
            infos[descriptor_idx].palette_len =
                self.palette_lens.get(orig_index).copied().unwrap_or(0);
        }

        if let Some(transform) = ground_transform_opt {
            let mut ground_info = GridInfo {
                resolution: self.ground_resolution,
                dim_x: self.ground_dims.0,
                dim_y: self.ground_dims.1,
                dim_z: self.ground_dims.2,
                storage_w: self.ground_storage.0,
                storage_h: self.ground_storage.1,
                storage_d: self.ground_storage.2,
                palette_base: self.ground_palette_base,
                palette_len: self.ground_palette_len,
                grid_to_world: transform,
                ..GridInfo::default()
            };
            ground_info.tile_payload_base = payloads_gpu.len() as u32;
            ground_info.tile_payload_len = 0;
            ready.push(self.ground_voxel_view.clone());
            ready_colors.push(self.ground_color_view.clone());
            ready_masks.push(self.ground_tile_mask_view.clone());
            infos.push(ground_info);
        }

        if payloads_gpu.is_empty() {
            payloads_gpu.push(TilePayloadGpu::default());
        }
        self.tile_payload_buffer =
            create_tile_payload_buffer(memory_allocator.clone(), &payloads_gpu);

        let grid_info_buffer = create_grid_info_buffer(memory_allocator.clone(), &infos);
        self.active_voxel_grids = ready.len() as u32;
        self.voxel_set = build_voxel_descriptor_set(
            descriptor_set_allocator,
            render_pipeline,
            &ready,
            &ready_colors,
            &ready_masks,
            self.palette_buffer.clone(),
            self.tile_payload_buffer.clone(),
            grid_info_buffer,
        );
    }

    pub fn apply_random_rotation(
        &mut self, index: usize, descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        render_pipeline: &HotReloadComputePipeline, memory_allocator: Arc<StandardMemoryAllocator>,
    ) {
        if index >= self.models.len() {
            return;
        }
        self.ensure_transform_capacity();
        let mut rng = rand::thread_rng();
        let rotation = random_rotation_matrix(&mut rng);
        if index < self.grid_user_transforms.len() {
            self.grid_user_transforms[index] = rotation;
        }
        self.rebuild_descriptor_set(descriptor_set_allocator, render_pipeline, memory_allocator);
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
        if self.voxel_images_enabled {
            self.placeholder_view = create_empty_voxel_placeholder(
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                queue.clone(),
                placeholder_resolution,
            );
        }
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
        for ps in &mut self.palette_slices {
            *ps = None;
        }
        for tm in &mut self.tile_mask_views {
            *tm = None;
        }
        for tp in &mut self.tile_payloads {
            *tp = None;
        }
        for ts in &mut self.tile_stats {
            *ts = None;
        }
        self.palette_bases.fill(0);
        self.palette_lens.fill(0);
        let gi = [GridInfo {
            resolution: self.voxel_resolution,
            dim_x: self.voxel_resolution,
            dim_y: self.voxel_resolution,
            dim_z: self.voxel_resolution,
            storage_w: self.voxel_resolution / 4,
            storage_h: self.voxel_resolution / 4,
            storage_d: self.voxel_resolution / 8,
            palette_base: 0,
            palette_len: 0,
            grid_to_world: translation_to_mat4([0.0, 0.0, 0.0]),
            ..GridInfo::default()
        }];
        let grid_info_buffer = create_grid_info_buffer(memory_allocator.clone(), &gi);
        self.tile_payload_buffer =
            create_tile_payload_buffer(memory_allocator.clone(), &[TilePayloadGpu::default()]);
        self.voxel_set = build_voxel_descriptor_set(
            descriptor_set_allocator.clone(),
            render_pipeline,
            std::slice::from_ref(&self.placeholder_view),
            std::slice::from_ref(&self.placeholder_color_view),
            std::slice::from_ref(&self.placeholder_tile_mask_view),
            self.palette_buffer.clone(),
            self.tile_payload_buffer.clone(),
            grid_info_buffer,
        );
        let (tx, rx) = mpsc::channel();
        self.voxel_result_rx = rx;
        let generation_id = self.voxel_generation;
        let cancel_flag = self.voxel_cancel_flag.clone();
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
                    let palette_span: u8 = u8::MAX;
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
                            let compression =
                                if result.dim_x == 0 || result.dim_y == 0 || result.dim_z == 0 {
                                    crate::tile_compression::TileCompressionResult::default()
                                } else {
                                    crate::tile_compression::classify_tiles(
                                        &vox,
                                        (result.dim_x, result.dim_y, result.dim_z),
                                        (result.storage_w, result.storage_h, result.storage_d),
                                    )
                                };
                            let tile_mask = compression.mask_entries;
                            let tile_payloads: Vec<[u32; 4]> =
                                compression.payloads.into_iter().map(|p| p.occupancy).collect();
                            let tile_stats = compression.stats;
                            let _ = txc.send(VoxelJobMessage::Finished {
                                generation: gen_thread,
                                index: idx,
                                colors,
                                resolution: result.used_resolution,
                                dim_x: result.dim_x,
                                dim_y: result.dim_y,
                                dim_z: result.dim_z,
                                storage_w: result.storage_w,
                                storage_h: result.storage_h,
                                storage_d: result.storage_d,
                                tile_mask,
                                tile_payloads,
                                tile_stats,
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
                    if let Some((vox, _colors, logical, storage)) =
                        ply_to_voxels_with_progress(&path, res_for_grid, prog_cb)
                    {
                        let (dim_x, dim_y, dim_z) = logical;
                        let color_count = dim_x as usize * dim_y as usize * dim_z as usize;
                        let colors = vec![0u8; color_count];
                        let mut rng = seeded_rng_for_path(&path);
                        let palette_slice = vec![[
                            rng.gen_range(0.0..1.0),
                            rng.gen_range(0.0..1.0),
                            rng.gen_range(0.0..1.0),
                            1.0,
                        ]];
                        if !cancelled.load(Ordering::Relaxed) {
                            let compression = if dim_x == 0 || dim_y == 0 || dim_z == 0 {
                                crate::tile_compression::TileCompressionResult::default()
                            } else {
                                crate::tile_compression::classify_tiles(
                                    &vox,
                                    (dim_x, dim_y, dim_z),
                                    storage,
                                )
                            };
                            let tile_mask = compression.mask_entries;
                            let tile_payloads: Vec<[u32; 4]> =
                                compression.payloads.into_iter().map(|p| p.occupancy).collect();
                            let tile_stats = compression.stats;
                            let _ = txc.send(VoxelJobMessage::Finished {
                                generation: gen_thread,
                                index: idx,
                                colors,
                                resolution: res_for_grid,
                                dim_x: dim_x,
                                dim_y: dim_y,
                                dim_z: dim_z,
                                storage_w: storage.0,
                                storage_h: storage.1,
                                storage_d: storage.2,
                                tile_mask,
                                tile_payloads,
                                tile_stats,
                            });
                            let _ = txc.send(VoxelJobMessage::PaletteSlice {
                                generation: gen_thread,
                                index: idx,
                                colors: palette_slice,
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
