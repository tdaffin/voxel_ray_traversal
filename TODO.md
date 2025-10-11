I want to accelerate rendering when the number of voxels gets really large.
To that end I want to look at implementing a bounding volume hierarcy (bvh) to index the voxels.


Storage options: a brickmap, an octree, and a SVDAG.

Octree prototype implemented (2025-10-11):
 - CPU builds linear octree per grid (depth capped at 6) after voxelization.
 - Nodes concatenated into a single SSBO (binding=4) with per-grid offsets/metadata (binding=5).
 - Shader performs simple stack-based traversal; falls back to dense DDA if no octree present.
 - Leaf handling currently samples dense voxel to confirm hit; color = coord-based debug (improve to palette + normal).

Follow-up improvements:
 - Replace coord color with proper shading path (reuse SHADE mode logic) for octree leaves.
 - Order child push in octree traversal by ray direction for better early-outs.
 - Compress OctNode: pack first_child high bit for leaf flag to avoid sentinel and separate mask.
 - Consider building SVDAG for static models to reduce memory further.
 - Explore hybrid brick + octree (store 8x8x8 bricks at depth N, skip deeper subdivision).
 - GPU-side octree construction for dynamic voxel scenes.
 - Frustum / screen-space LOD: choose depth limit per ray to reduce steps for distant geometry.
 - Collect traversal statistics (steps saved vs dense) and expose in UI.
 - Handle non-power-of-two dims more tightly (current root bounds power-of-two cube).

Potential risks / TODO:
 - Stack size fixed (32). Increase or make iterative if deeper trees introduced.
 - Current leaf mask unused for early exit refinement; could store 2x2x2 occupancy to skip dense sample.
 - Need to ensure alignment / padding matches host (verify with sizeof if adding fields).
