use dot_vox::load;
use std::path::Path;

/// Convert a MagicaVoxel .vox file into packed voxel bitfield and palette indices.
/// Returns (occupancy_bitfield_vec_u128, color_indices_per_voxel)
/// base_palette_index: starting index in global palette where this model's colors will be placed.
/// palette_span: maximum number of palette slots reserved for this model.
/// Returns (voxels, remapped_color_indices, palette_colors_used)
pub fn vox_to_voxels(
    path: impl AsRef<Path>, target_resolution: u32, base_palette_index: u8, palette_span: u8,
) -> Option<(Vec<u128>, Vec<u8>, Vec<[f32; 4]>)> {
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

    let res = target_resolution as usize;
    if res == 0 {
        return None;
    }
    let mut voxels = vec![0u128; res * res * res / 128];
    let mut colors = vec![0u8; res * res * res];

    let scale = (res as f32 - 1.0) / ((sx.max(sy)).max(sz)) as f32;

    // Build a remap table from MagicaVoxel color index -> local subrange offset
    // MagicaVoxel palette indices are 1..=255; 0 unused.
    let mut remap: [u8; 256] = [0; 256];
    let mut used_colors: Vec<[f32; 4]> = Vec::new();
    // Prepare palette slice (truncate to palette_span)
    // scene.palette holds 256 RGBA (u8) entries (MagicaVoxel), default if missing.
    let palette_rgba = &scene.palette;
    // We'll lazily assign as voxels encountered to keep only used subset (up to span)

    for v in &model.voxels {
        let x = (v.x as f32 * scale).round() as i32;
        let y = (v.y as f32 * scale).round() as i32;
        let z = (v.z as f32 * scale).round() as i32;
        if x < 0 || y < 0 || z < 0 || x >= res as i32 || y >= res as i32 || z >= res as i32 {
            continue;
        }
        let (x, y, z) = (x as usize, y as usize, z as usize);
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
    Some((voxels, colors, used_colors))
}
