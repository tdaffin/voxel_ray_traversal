use dot_vox::load;
use std::path::Path;

/// Convert a MagicaVoxel .vox file into packed voxel bitfield and palette indices.
/// Returns (occupancy_bitfield_vec_u128, color_indices_per_voxel)
/// base_palette_index: starting index in global palette where this model's colors will be placed.
/// palette_span: maximum number of palette slots reserved for this model.
/// Returns (voxels, remapped_color_indices, palette_colors_used)
pub struct VoxLoadResult {
    pub voxels: Vec<u128>,
    pub colors: Vec<u8>,
    pub palette: Vec<[f32; 4]>,
    pub used_resolution: u32, // legacy cubic resolution (kept for compatibility)
    pub dim_x: u32,           // logical X dimension
    pub dim_y: u32,           // logical Y dimension
    pub dim_z: u32,           // logical Z dimension
    pub storage_w: u32,       // packed texel width (ceil(dim_x/4))
    pub storage_h: u32,       // packed texel height (ceil(dim_y/4))
    pub storage_d: u32,       // packed texel depth (ceil(dim_z/8))
}

pub fn vox_to_voxels(
    path: impl AsRef<Path>,
    target_resolution: Option<u32>,
    palette_span: u8, // maximum colors to keep (local palette length cap)
) -> Option<VoxLoadResult> {
    #[allow(dead_code)]
    const _VOX_LOADER_VERSION: &str = "vox_loader_v1";
    let path_ref = path.as_ref();
    let path_str = path_ref.to_str()?; // return None if non-UTF8
    let scene = load(path_str).ok()?;
    let model = scene.models.get(0)?; // first model only

    let sx = model.size.x as usize;
    let sy = model.size.y as usize;
    let sz = model.size.z as usize;
    if sx == 0 || sy == 0 || sz == 0 {
        return None;
    }

    let native_max = (sx.max(sy)).max(sz) as u32;
    let requested = target_resolution.unwrap_or(native_max).max(native_max);
    // For now keep used_resolution for backwards compatibility (still padded cubic) but allocate non-cubic storage below.
    let mut cubic = requested;
    if cubic % 8 != 0 {
        cubic += 8 - (cubic % 8);
    }
    let used_resolution = cubic; // legacy value
    // Non-cubic packed storage extents (texel units are 4x4x8 voxels)
    let storage_w = ((sx as u32) + 3) / 4;
    let storage_h = ((sy as u32) + 3) / 4;
    let storage_d = ((sz as u32) + 7) / 8;
    let packed_texel_count = (storage_w * storage_h * storage_d) as usize;
    let mut voxels = vec![0u128; packed_texel_count];
    let mut colors = vec![0u8; sx * sy * sz];

    // Build a remap table from MagicaVoxel color index -> local subrange offset
    // MagicaVoxel palette indices are 1..=255; 0 unused.
    let mut remap: [u8; 256] = [0; 256];
    let mut remap_assigned: [bool; 256] = [false; 256];
    let mut used_colors: Vec<[f32; 4]> = Vec::new();
    // Determine maximum number of palette entries to capture for this model.
    let max_colors = if palette_span == 0 { 255 } else { palette_span as usize };
    // Prepare palette slice (truncate to palette_span, if provided)
    // scene.palette holds 256 RGBA (u8) entries (MagicaVoxel), default if missing.
    let palette_rgba = &scene.palette;
    // We'll lazily assign as voxels encountered to keep only used subset (up to span)

    for v in &model.voxels {
        // Preserve original coordinates (no isotropic scaling). Place directly into padded cubic volume.
        let x = v.x as usize;
        let y = v.y as usize;
        let z = v.z as usize;
        // Bounds already guaranteed within logical dims
        let tx = x / 4;
        let ty = y / 4;
        let tz = z / 8;
        let texel = (tx + (ty + tz * storage_h as usize) * storage_w as usize) as usize;
        let bit = (x % 4) * 32 + (y % 4) + (z % 8) * 4;
        voxels[texel] |= 1u128 << bit;
        let orig = v.i as usize; // 0..255
        let mapped_local = if orig == 0 {
            0
        } else if remap_assigned[orig] {
            remap[orig]
        } else if used_colors.len() < max_colors {
            let pal = palette_rgba[orig];
            let rgba = [
                pal.r as f32 / 255.0,
                pal.g as f32 / 255.0,
                pal.b as f32 / 255.0,
                pal.a as f32 / 255.0,
            ];
            let local_index = used_colors.len() as u8;
            used_colors.push(rgba);
            remap[orig] = local_index;
            remap_assigned[orig] = true;
            local_index
        } else {
            // Fallback: reuse the first color if available, otherwise zero.
            remap[orig] = 0;
            remap_assigned[orig] = true;
            0
        };
        colors[(z * sy + y) * sx + x] = mapped_local;
    }
    Some(VoxLoadResult {
        voxels,
        colors,
        palette: used_colors,
        used_resolution: used_resolution,
        dim_x: sx as u32,
        dim_y: sy as u32,
        dim_z: sz as u32,
        storage_w,
        storage_h,
        storage_d,
    })
}
