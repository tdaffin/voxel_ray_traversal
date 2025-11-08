# 4×4×8 Tile Compression Plan

## Goals
- Reduce GPU-side voxel storage for empty and uniform 4×4×8 tiles without regressing traversal performance.
- Maintain compatibility with existing packed occupancy layout for non-uniform tiles.
- Keep data generation changes localized to Rust voxelization/upload code and shader sampling logic.

## High-Level Approach
1. Introduce a tile mask structure that records, per 4×4×8 region, whether the tile is empty, uniform, or needs the packed payload.
2. Store payload data for non-uniform tiles only in a dense buffer; keep palette indices alongside occupancy bits.
3. Teach the compute shader to check the tile mask before reading packed data, short-circuiting for empty/uniform tiles.
4. Update voxel asset ingestion to emit the new tile mask and payload buffers.

## Data Format Changes
- **Tile mask image (new):**
  - `r32ui` 3D texture sized `(grid_dims / (4,4,8))`.
  - Bit layout proposal:
    - Bits 0-29: payload index (0-based) for non-uniform tiles.
    - Bit 30: uniform-flag (1 = uniform tile, payload index encodes palette/material id; 0 = non-uniform).
    - Bit 31: empty-flag (1 = empty tile, remaining bits ignored).
  - Reserved indices 0/1 for sentinel values if needed; adjust host-side offsets accordingly.
  - Binding plan: set `1`, binding `4` (matches reserved constants in `voxel.rs`).
- **Tile payload buffer (new SSBO):**
  - Array of structs containing:
    - `uvec4 occupancy_words;` // four 32-bit lanes, same as current packed texel.
    - `uvec4 palette_words;` // optional if palette indices remain image-based; include only if helps coherence.
  - Only populated for non-uniform tiles.
  - Binding plan: set `1`, binding `5`.
- **Existing occupancy image:**
  - Becomes optional; retained temporarily for incremental rollout and fallback.

## Host-Side Tasks (Rust)
1. **Voxelizer adjustments:**
   - Iterate voxels in 4×4×8 tiles during build.
   - Classify each tile: empty, uniform (single palette id + occupied voxels), mixed.
   - Emit mask entry + optional dense payload.
2. **Buffer creation:**
   - Allocate `Image3D` for the tile mask (`r32ui`, descriptor set 1 binding TBD).
   - Allocate SSBO for tile payload data; consider 16-byte alignment.
   - Update descriptor set layouts, pipelines, and binding code (`pipelines.rs`, `app_builder.rs`).
3. **Asset versioning:**
   - Add metadata or version tag so runtime knows whether a grid uses compressed tiles or legacy layout.
   - Provide conversion path or rebuild assets.
4. **Fallback path:**
   - For small grids or debugging, allow forcing legacy layout (env flag or config).

## Shader Tasks (`shaders/traverse.comp`)
1. Add bindings for the tile mask image and payload SSBO.
2. Update `readVoxel` to:
   - Compute tile coordinate (`coord >> (2,2,3)` style).
   - Fetch mask entry.
   - If empty flag set -> return `false`.
   - If uniform flag set -> return stored bit/palette value without SSBO access.
   - Else read payload via index, preserving existing caching logic (`texel` + `texel_coord`).
3. Ensure `always_instant` and other push constants still function with new fast paths.
4. Compile-time guard or specialization constant to support legacy mode until rollout complete.

## Engine Integration
- Extend `GridInfo` to include offsets/strides for new resources.
- Adjust hot-reload / asset streaming paths to upload tile mask + payload buffers.
- Update `voxel_facade.rs` / `voxel_job.rs` as needed to supply new data to GPU.

## Performance Validation
1. Benchmark traversal shader before/after using representative scenes (dense & sparse).
2. Measure GPU memory usage change per grid.
3. Profile CPU voxelization time; ensure tile classification does not dominate build.
4. Watch for cache misses or divergence introduced by new branches; optimize mask layout if needed.

## Testing Checklist
- Unit tests for tile classification (empty/uniform/mixed) and mask encoding.
- Integration test that renders known sparse scene and matches reference image.
- Shader validation with RenderDoc / spirv-dis to confirm no unexpected instructions.
- Regression test for legacy assets.

## Open Questions / Next Steps
- Decide whether uniform tiles store palette id only or also occupancy bit (for partially filled uniform material).
- Determine maximum palette ids to ensure mask entry has enough bits.
- Explore compressing even uniform occupancy (e.g., 0 or 1 bit) to save mask bits.
- Consider GPU-side generation path using compute preprocessing in future iterations.
