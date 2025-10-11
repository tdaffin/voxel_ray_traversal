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
    pub native_resolution: u32, // original largest source dimension (pre padding)
    pub used_resolution: u32,   // storage resolution used (padded to multiples for packing)
    pub dim_x: u32,             // original X dimension (no isotropic scaling applied)
    pub dim_y: u32,             // original Y dimension
    pub dim_z: u32,             // original Z dimension
}

pub fn vox_to_voxels(
    path: impl AsRef<Path>, target_resolution: Option<u32>, base_palette_index: u8,
    palette_span: u8,
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

    let native_max = (sx.max(sy)).max(sz) as u32; // original largest dimension
    let requested = target_resolution.unwrap_or(native_max);
    // Storage resolution (cubic) – round up to multiple of 8 for packing (z/8) & 4 for x,y/4.
    let mut chosen = requested.max(native_max);
    if chosen == 0 {
        return None;
    }
    if chosen % 8 != 0 {
        chosen += 8 - (chosen % 8);
    }
    let res = chosen as usize;
    let mut voxels = vec![0u128; res * res * res / 128];
    let mut colors = vec![0u8; res * res * res];

    // Build a remap table from MagicaVoxel color index -> local subrange offset
    // MagicaVoxel palette indices are 1..=255; 0 unused.
    let mut remap: [u8; 256] = [0; 256];
    let mut used_colors: Vec<[f32; 4]> = Vec::new();
    // Prepare palette slice (truncate to palette_span)
    // scene.palette holds 256 RGBA (u8) entries (MagicaVoxel), default if missing.
    let palette_rgba = &scene.palette;
    // We'll lazily assign as voxels encountered to keep only used subset (up to span)

    for v in &model.voxels {
        // Preserve original coordinates (no isotropic scaling). Place directly into padded cubic volume.
        let x = v.x as usize;
        let y = v.y as usize;
        let z = v.z as usize;
        if x >= res || y >= res || z >= res {
            continue;
        }
        let texel = (x + ((y + (z / 8) * res) / 4) * res) / 4;
        let bit = (x % 4) * 32 + (y % 4) + (z % 8) * 4;
        voxels[texel] |= 1u128 << bit;
        let orig = v.i as usize; // 0..255
        let mapped = if remap[orig] != 0 || orig == 0 {
            remap[orig]
        } else {
            // Need to allocate new slot in local subrange
            if used_colors.len() < palette_span as usize {
                let pal = palette_rgba[orig];
                let rgba = [
                    pal.r as f32 / 255.0,
                    pal.g as f32 / 255.0,
                    pal.b as f32 / 255.0,
                    pal.a as f32 / 255.0,
                ];
                used_colors.push(rgba);
                let local_offset = used_colors.len() as u8 - 1;
                let global_index = base_palette_index.saturating_add(local_offset);
                remap[orig] = global_index.max(1); // keep 0 reserved if orig==0
                remap[orig]
            } else {
                // Subrange full: reuse first slot (could be improved with nearest-color mapping)
                base_palette_index
            }
        };
        colors[(z * res + y) * res + x] = mapped;
    }
    Some(VoxLoadResult {
        voxels,
        colors,
        palette: used_colors,
        native_resolution: native_max,
        used_resolution: chosen,
        dim_x: sx as u32,
        dim_y: sy as u32,
        dim_z: sz as u32,
    })
}
