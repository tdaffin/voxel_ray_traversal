use std::vec::Vec;

/// Compact linear octree node representation.
/// Each internal node stores the index of its first child in the linear array; the 8 children
/// occupy consecutive slots [first_child..first_child+8).
/// Leaves set `first_child = u32::MAX` and `mask` encodes 2x2x2 occupancy bits (optional future use).
/// For now `mask` just stores which child voxels were occupied at build time so that very small
/// leaf bricks (depth==max_depth) can still test exact voxel occupancy without sampling the full
/// 3D texture.
use vulkano::buffer::BufferContents;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, BufferContents)]
pub struct OctNode {
    pub first_child: u32, // u32::MAX indicates leaf
    pub mask: u8,         // occupancy mask for leaf (8 bits child order: (x|y<<1|z<<2))
    pub _pad: [u8; 3],    // explicit padding for std140-like alignment
}

/// Per-grid octree metadata uploaded alongside existing GridInfo.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, BufferContents)]
pub struct OctreeGridInfo {
    pub node_offset: u32, // starting index into global octree node buffer
    pub node_count: u32,  // number of nodes for this grid
    pub max_depth: u32,   // maximum depth (root depth = 0)
    pub flags: u32,       // bit0: 1 if octree present
}

/// Build a linear octree from a dense voxel occupancy accessor.
///
/// `dims`: logical voxel dimensions (x,y,z)
/// `is_solid`: closure returning true if voxel (x,y,z) is occupied
/// `max_depth_limit`: optional cap to avoid over-refining (default = auto based on dims)
/// Returns (nodes, max_depth_reached).
pub fn build_octree<F>(
    dims: (u32, u32, u32), mut is_solid: F, max_depth_limit: Option<u32>,
) -> (Vec<OctNode>, u32)
where
    F: FnMut(u32, u32, u32) -> bool,
{
    let (dx, dy, dz) = dims;
    let max_dim = dx.max(dy).max(dz);
    // Depth needed so that 2^depth >= max_dim
    let mut target_depth = 0u32;
    while (1u32 << target_depth) < max_dim {
        target_depth += 1;
    }
    if let Some(limit) = max_depth_limit {
        target_depth = target_depth.min(limit);
    }

    #[derive(Clone, Copy)]
    struct StackEntry {
        node_index: u32,
        depth: u32,
        ox: u32,
        oy: u32,
        oz: u32,
        size: u32,
    }

    let mut nodes: Vec<OctNode> = Vec::new();
    // push root
    nodes.push(OctNode { first_child: u32::MAX, mask: 0, _pad: [0; 3] });
    let mut stack = vec![StackEntry {
        node_index: 0,
        depth: 0,
        ox: 0,
        oy: 0,
        oz: 0,
        size: 1u32 << target_depth,
    }];
    let mut max_depth_reached = 0u32;

    while let Some(entry) = stack.pop() {
        max_depth_reached = max_depth_reached.max(entry.depth);
        let size = entry.size;
        let leaf = entry.depth == target_depth || size == 1;
        if leaf {
            // compute occupancy mask over up to 2x2x2 voxels (if size>1 we collapse region)
            let half = size.min(2); // sample at up to 2 resolution for mask
            let mut mask = 0u8;
            for z in 0..half {
                for y in 0..half {
                    for x in 0..half {
                        let gx = entry.ox + x;
                        let gy = entry.oy + y;
                        let gz = entry.oz + z;
                        if gx < dx && gy < dy && gz < dz && is_solid(gx, gy, gz) {
                            let bit = x | (y << 1) | (z << 2);
                            mask |= 1 << bit;
                        }
                    }
                }
            }
            nodes[entry.node_index as usize].mask = mask;
            continue;
        }
        // Determine if region is empty or full
        let half = size / 2;
        let mut any = false;
        let mut all = true;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let gx = entry.ox + x;
                    let gy = entry.oy + y;
                    let gz = entry.oz + z;
                    if gx >= dx || gy >= dy || gz >= dz {
                        all = false;
                        continue;
                    }
                    let occ = is_solid(gx, gy, gz);
                    any |= occ;
                    all &= occ;
                }
            }
        }
        if !any {
            // empty leaf
            nodes[entry.node_index as usize].mask = 0; // empty
            continue;
        }
        if all {
            // full leaf
            nodes[entry.node_index as usize].mask = 0xFF; // full
            continue;
        }
        // Subdivide
        let first_child = nodes.len() as u32;
        nodes[entry.node_index as usize].first_child = first_child;
        // allocate 8 children placeholders
        for _ in 0..8 {
            nodes.push(OctNode { first_child: u32::MAX, mask: 0, _pad: [0; 3] });
        }
        for cz in 0..2 {
            for cy in 0..2 {
                for cx in 0..2 {
                    let child_index = first_child + (cx | (cy << 1) | (cz << 2)) as u32;
                    let ox = entry.ox + cx * half;
                    let oy = entry.oy + cy * half;
                    let oz = entry.oz + cz * half;
                    stack.push(StackEntry {
                        node_index: child_index,
                        depth: entry.depth + 1,
                        ox,
                        oy,
                        oz,
                        size: half,
                    });
                }
            }
        }
    }

    (nodes, max_depth_reached)
}
