use dot_vox::load;
use std::path::Path;

/// Convert a MagicaVoxel .vox file into packed voxel bitfield and palette indices.
/// Returns (occupancy_bitfield_vec_u128, color_indices_per_voxel)
pub fn vox_to_voxels(
    path: impl AsRef<Path>, target_resolution: u32,
) -> Option<(Vec<u128>, Vec<u8>)> {
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
        colors[(z * res + y) * res + x] = v.i;
    }

    // TODO: Integrate scene.palette into palette buffer or remap indices.
    Some((voxels, colors))
}
