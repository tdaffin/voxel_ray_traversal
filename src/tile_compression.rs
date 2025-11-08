/// Bit flag marking a tile entry as empty.
pub const TILE_EMPTY_BIT: u32 = 1 << 31;
/// Bit flag marking a tile entry as uniform (single occupancy value encoded in the low bits).
pub const TILE_UNIFORM_BIT: u32 = 1 << 30;
/// When `TILE_UNIFORM_BIT` is set, this bit encodes the occupancy value (0 or 1).
pub const TILE_UNIFORM_VALUE_BIT: u32 = 1 << 0;
/// Mask selecting the payload index bits for non-uniform tiles (bits 0-29).
pub const TILE_PAYLOAD_INDEX_MASK: u32 = 0x3FFF_FFFF;

/// Four 32-bit words matching the RGBA lanes of a packed 4×4×8 occupancy texel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TilePayload {
    pub occupancy: [u32; 4],
}

impl TilePayload {
    pub fn from_u128(bits: u128) -> Self {
        Self {
            occupancy: [
                (bits & 0xFFFF_FFFFu128) as u32,
                ((bits >> 32) & 0xFFFF_FFFFu128) as u32,
                ((bits >> 64) & 0xFFFF_FFFFu128) as u32,
                ((bits >> 96) & 0xFFFF_FFFFu128) as u32,
            ],
        }
    }
}

/// Summary of how tiles were classified for compression statistics and debugging.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileCompressionStats {
    pub empty_tiles: usize,
    pub uniform_tiles: usize,
    pub dense_tiles: usize,
}

/// Result of classifying a voxel grid into sparse tiles.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileCompressionResult {
    pub mask_entries: Vec<u32>,
    pub payloads: Vec<TilePayload>,
    pub stats: TileCompressionStats,
}

/// Classify 4×4×8 packed tiles and build mask/payload arrays.
///
/// * `occupancy_blocks` must be ordered in the same layout as the GPU image upload (x-major).
/// * `dims` is the logical voxel dimensions; valid voxels outside these bounds are ignored.
/// * `storage` is the packed tile extents (ceil divisons of dims by 4/4/8).
///
/// The returned mask has one entry per tile. Empty tiles set `TILE_EMPTY_BIT`. Uniform tiles set
/// `TILE_UNIFORM_BIT` and encode the occupancy value in `TILE_UNIFORM_VALUE_BIT`. Non-uniform tiles
/// encode the dense payload index (0-based) in the low bits.
pub fn classify_tiles(
    occupancy_blocks: &[u128], dims: (u32, u32, u32), storage: (u32, u32, u32),
) -> TileCompressionResult {
    let (storage_w, storage_h, storage_d) = storage;
    assert_eq!(
        occupancy_blocks.len(),
        storage_w as usize * storage_h as usize * storage_d as usize,
        "packed tile count does not match storage extents"
    );

    let mut mask_entries = Vec::with_capacity(occupancy_blocks.len());
    let mut payloads = Vec::new();
    let mut stats = TileCompressionStats::default();

    for (tile_index, &raw_bits) in occupancy_blocks.iter().enumerate() {
        let tile_coord = index_to_tile_coord(tile_index as u32, storage);
        let valid_mask = valid_mask_for_tile(tile_coord, dims);
        if valid_mask == 0 {
            mask_entries.push(TILE_EMPTY_BIT);
            stats.empty_tiles += 1;
            continue;
        }

        let sanitized_bits = raw_bits & valid_mask;
        if sanitized_bits == 0 {
            mask_entries.push(TILE_EMPTY_BIT);
            stats.empty_tiles += 1;
            continue;
        }

        if sanitized_bits == valid_mask {
            mask_entries.push(TILE_UNIFORM_BIT | TILE_UNIFORM_VALUE_BIT);
            stats.uniform_tiles += 1;
            continue;
        }

        let payload_index = payloads.len();
        assert!(payload_index <= TILE_PAYLOAD_INDEX_MASK as usize);
        payloads.push(TilePayload::from_u128(sanitized_bits));
        mask_entries.push(payload_index as u32 & TILE_PAYLOAD_INDEX_MASK);
        stats.dense_tiles += 1;
    }

    TileCompressionResult { mask_entries, payloads, stats }
}

fn index_to_tile_coord(index: u32, storage: (u32, u32, u32)) -> (u32, u32, u32) {
    let (storage_w, storage_h, _storage_d) = storage;
    let tiles_per_layer = storage_w * storage_h;
    let z = index / tiles_per_layer;
    let rem = index % tiles_per_layer;
    let y = rem / storage_w;
    let x = rem % storage_w;
    (x, y, z)
}

fn valid_mask_for_tile(tile_coord: (u32, u32, u32), dims: (u32, u32, u32)) -> u128 {
    let (tile_x, tile_y, tile_z) = tile_coord;
    let (dim_x, dim_y, dim_z) = dims;
    let start_x = tile_x * 4;
    let start_y = tile_y * 4;
    let start_z = tile_z * 8;
    let mut mask = 0u128;

    for local_x in 0..4 {
        let gx = start_x + local_x;
        if gx >= dim_x {
            break;
        }
        for local_z in 0..8 {
            let gz = start_z + local_z;
            if gz >= dim_z {
                break;
            }
            let z_stride = local_z * 4;
            for local_y in 0..4 {
                let gy = start_y + local_y;
                if gy >= dim_y {
                    break;
                }
                let bit_index = local_x * 32 + z_stride + local_y;
                mask |= 1u128 << bit_index;
            }
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_empty_tile() {
        let dims = (4, 4, 8);
        let storage = (1, 1, 1);
        let result = classify_tiles(&[0u128], dims, storage);
        assert_eq!(result.mask_entries, vec![TILE_EMPTY_BIT]);
        assert!(result.payloads.is_empty());
        assert_eq!(result.stats.empty_tiles, 1);
        assert_eq!(result.stats.uniform_tiles, 0);
        assert_eq!(result.stats.dense_tiles, 0);
    }

    #[test]
    fn classify_uniform_tile() {
        let dims = (4, 4, 8);
        let storage = (1, 1, 1);
        let full = u128::MAX;
        let result = classify_tiles(&[full], dims, storage);
        assert_eq!(result.mask_entries, vec![TILE_UNIFORM_BIT | TILE_UNIFORM_VALUE_BIT]);
        assert!(result.payloads.is_empty());
        assert_eq!(result.stats.empty_tiles, 0);
        assert_eq!(result.stats.uniform_tiles, 1);
        assert_eq!(result.stats.dense_tiles, 0);
    }

    #[test]
    fn classify_partial_tile() {
        let dims = (3, 3, 3);
        let storage = (1, 1, 1);
        let valid_mask = valid_mask_for_tile((0, 0, 0), dims);
        let result = classify_tiles(&[valid_mask], dims, storage);
        assert_eq!(result.mask_entries, vec![TILE_UNIFORM_BIT | TILE_UNIFORM_VALUE_BIT]);
        assert!(result.payloads.is_empty());
        assert_eq!(result.stats.uniform_tiles, 1);
    }

    #[test]
    fn classify_dense_tile() {
        let dims = (4, 4, 8);
        let storage = (1, 1, 1);
        let mut bits = 0u128;
        bits |= 1u128 << 0; // (0,0,0)
        bits |= 1u128 << 63; // arbitrary interior bit
        let result = classify_tiles(&[bits], dims, storage);
        assert_eq!(result.mask_entries.len(), 1);
        assert_eq!(result.mask_entries[0] & TILE_PAYLOAD_INDEX_MASK, 0);
        assert_eq!(result.payloads.len(), 1);
        assert_eq!(result.payloads[0], TilePayload::from_u128(bits));
        assert_eq!(result.stats.dense_tiles, 1);
    }

    #[test]
    fn classify_multiple_tiles() {
        let dims = (8, 4, 8);
        let storage = (2, 1, 1);
        let blocks = [0u128, u128::MAX];
        let result = classify_tiles(&blocks, dims, storage);
        assert_eq!(result.mask_entries.len(), 2);
        assert_eq!(result.mask_entries[0], TILE_EMPTY_BIT);
        assert_eq!(result.mask_entries[1], TILE_UNIFORM_BIT | TILE_UNIFORM_VALUE_BIT);
        assert!(result.payloads.is_empty());
    }

    #[test]
    fn dense_tile_strips_invalid_bits() {
        let dims = (5, 4, 8);
        let storage = (2, 1, 1);
        let mut raw = u128::MAX;
        raw &= !(1u128 << 0); // clear one valid bit to avoid uniform classification
        // second tile partially exceeds dims along X; ensure invalid bits are dropped
        let result = classify_tiles(&[0u128, raw], dims, storage);
        assert_eq!(result.payloads.len(), 1);
        let payload_bits = &result.payloads[0].occupancy;
        // Reconstruct u128 and validate only in-range bits kept
        let sanitized = (payload_bits[0] as u128)
            | ((payload_bits[1] as u128) << 32)
            | ((payload_bits[2] as u128) << 64)
            | ((payload_bits[3] as u128) << 96);
        let valid_mask = valid_mask_for_tile((1, 0, 0), dims);
        assert_ne!(raw & !valid_mask, 0); // raw contained out-of-range bits
        assert_eq!(sanitized & !valid_mask, 0);
        assert_eq!(sanitized, valid_mask & raw);
    }
}
