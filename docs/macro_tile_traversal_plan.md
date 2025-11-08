# Macro Tile Traversal Optimization Plan

## Goal
Speed up voxel grid traversal by leveraging the existing 4×4×8 tile masks to skip large empty regions, reducing the number of fine-grained voxel steps and memory accesses.

## Current State
- Compute shader `traverse.comp` performs fine-grained 3D DDA per voxel.
- `readVoxel` reads occupancy via `tileMaskImages` and dense payload buffer, but every step still advances by one voxel even when entire tiles are empty.
- Tile metadata exposed on the GPU:
  - `tileMaskImages[g]` holds one `uint` per 4×4×8 tile with flags for empty/uniform/dense.
  - `GridInfo` contains per-grid dimensions and packed storage extents.

## Proposed Optimization Overview
Introduce a two-level traversal:
1. **Coarse DDA** over the tile grid (step size 4×4×8 voxels) using tile masks.
2. **Fine DDA** inside tiles marked non-empty (uniform or dense), using existing `readVoxel` logic.

When a tile is classified as empty the coarse traversal jumps directly to the next tile boundary, skipping the 128 voxel steps that would otherwise occur. For uniform tiles we can record the occupancy once and either:
- Skip fine traversal entirely when the uniform value is zero (empty), or
- Enter fine traversal with a fast path that immediately reports a hit when the uniform value is one.

## Detailed Tasks

### 1. Data Preparation & Helpers
- [ ] Add helper in shader to fetch tile metadata (mask word) without altering cache state.
- [ ] Extend `GridInfo` with tile grid extents if needed (currently inferred as `storageW/H/D`).
- [ ] Document mask bit semantics in shader for clarity.

### 2. Coarse Tile DDA
- [ ] Implement per-grid setup computing tile-space ray origin/direction:
  - Convert entry point and direction from voxel space to tile coordinates.
  - Precompute reciprocal direction to avoid repeated division.
- [ ] Run a tile-level DDA loop advancing by whole tiles until:
  - Ray exits tile bounds (no hit) or
  - Tile mask indicates potential occupancy (uniform=1 or dense payload).
- [ ] Handle tiles marked empty by advancing to `tNextTile`.

### 3. Tile Classification Handling
- [ ] **Uniform empty**: treat same as empty tile and continue coarse traversal.
- [ ] **Uniform filled**: register a hit immediately or optionally perform minimal fine traversal to gather normal/hit position.
- [ ] **Dense tile**: switch to existing fine-grained traversal, but limit stepping to the tile’s local bounds to avoid crossing tile boundary before re-checking masks.

### 4. Fine Traversal Integration
- [ ] Modify current fine DDA to accept tile bounds and exit when leaving the current tile.
- [ ] When fine traversal exits without hit, resume coarse traversal from the next tile.
- [ ] Ensure `readVoxel` continues to cache the dense payload for repeated lookups within the tile.

### 5. Edge Cases & Correctness
- [ ] Validate behavior for rays starting inside non-empty tiles (skip coarse step, go straight to fine traversal).
- [ ] Handle rays aligned with axes (avoid division by zero in coarse DDA).
- [ ] Ensure normals and hit positions are computed correctly when hits occur at tile boundaries.
- [ ] Preserve `always_instant` debug path semantics.

### 6. Performance Instrumentation
- [ ] Add optional debug counters (e.g., number of coarse vs fine steps) guarded by verbose logging or compile flag for benchmarking.
- [ ] Measure traversal performance on representative scenes before/after optimization.

### 7. Host-Side Support (if required)
- [ ] No host changes expected, but confirm tile extents (`storageW/H/D`) remain accurate.
- [ ] Update documentation (`block_tile_compression_plan.md`) referencing new traversal behavior.

### 8. Testing Strategy
- [ ] GPU shader unit tests (if available) or synthetic scenes to verify hit/miss accuracy.
- [ ] Compare rendered frames before/after for visual regressions.
- [ ] Run benchmarking hook to validate performance gains.

## Risks & Mitigations
- **Precision issues** when converting between voxel and tile space: use consistent floor/ceil logic and guard against floating-point error.
- **Control flow complexity** in shader may increase register pressure; profile and simplify as needed.
- **Uniform tile hit handling** must preserve correct normal and hit location—consider deriving from entry face normals.
- **Regression in dense scenes** where coarse traversal overhead could offset gains; keep coarse loop lightweight and exit early when unnecessary.

## Success Criteria
- Demonstrable reduction in average fine voxel steps for sparse scenes.
- Equal or better frame times compared to baseline across key models.
- Identical rendered output (within expected floating-point tolerances).

## Follow-Up Enhancements (Stretch Goals)
- Cache last non-empty tile to avoid redundant mask fetches per fine step.
- Support multi-resolution tile hierarchies (e.g., 8×8×16 tiles) if compression metadata expands.
- Integrate statistics into UI to monitor coarse step effectiveness.
