# Distance-Aware Camera Zoom

Goal: when the user scroll-zooms, translate the camera forward only up to the first non-empty voxel along the view ray instead of using a fixed step. This avoids tunneling into models and keeps zoom feel consistent regardless of scene scale.

## Current Behavior
- `InputController::update` (src/input_controller.rs) reads the mouse wheel delta and always moves the camera a fixed amount (`zoom_speed * scroll_delta`) along the forward vector produced by `camera.rotation_matrix()`.
- No depth tests happen on the CPU side, so the camera can pierce through voxel surfaces.

## Simplest Path to Distance-Aware Zoom
1. **Surface data availability**
   - During voxelization (`voxel_job.rs`), we already receive CPU-side tile mask words (`Vec<u32>`) and dense payloads (`Vec<[u32; 4]>`) for every grid before uploading them to images/buffers.
   - Persist those raw vectors in a lightweight struct (e.g. `GridOccupancyCache`) alongside `grid_dims`, `grid_storage`, and the grid's `GridInfo::grid_to_world` matrix.
   - Store the cache inside `VoxelManager` (mirroring how `tile_payloads` are kept) and expose read-only access via a method such as `VoxelManager::grid_occupancy(idx) -> Option<&GridOccupancyCache>`.

2. **Ray/AABB prep per grid**
   - Use each grid's `grid_to_world` matrix (already filled when descriptor sets are rebuilt) and compute its inverse once, caching `world_to_grid` to avoid per-zoom inverses.
   - Treat the logical voxel bounds as the axis-aligned box `[0, dim_x) × [0, dim_y) × [0, dim_z)` in grid space.
   - For every zoom tick, transform the camera origin and normalized forward vector into the grid's local space with `world_to_grid`.
   - Run a standard ray vs. AABB intersection. Skip grids that the ray misses outright.

3. **Voxel DDA inside a grid**
   - After entering a grid, perform a 3D DDA (digital differential analyzer) stepping one voxel at a time. The tile compression already partitions space into 4×4×8 chunks; however, for simplicity we can expand occupancy to a `BitVec` (or lazily query tiles) and treat it as dense.
   - To avoid decompressing every time, convert mask+payload into a `Vec<u64>` or `BitVec` when the grid cache is built. A helper like `GridOccupancyCache::is_filled(voxel_ijk: [u32;3]) -> bool` can translate the voxel coordinate to the proper tile and bit by following the same math as `tile_compression.rs`.
   - Pseudocode sketch:

```rust
fn march_to_hit(ray_origin: Vec3, ray_dir: Vec3, grid: &GridOccupancyCache) -> Option<f64> {
    let mut cell = start_voxel(ray_origin);
    let mut t = 0.0;
    while cell.inside(grid.dims) && t < grid.max_distance {
        if grid.is_filled(cell) {
            return Some(t);
        }
        let (step_axis, next_t) = advance_dda(ray_origin, ray_dir, cell);
        cell[step_axis] += step_dir(step_axis);
        t = next_t;
    }
    None
}
```
   - Convert the resulting `t` (in grid-space units) back to world units by applying the scale encoded in `grid_to_world`. Because grids may be scaled non-uniformly, compute distance using the world-space entry/exit points: transform both `ray_origin` and `ray_origin + ray_dir * t` back out and measure the Euclidean distance of those vectors in world space.

4. **Aggregating across grids**
   - Iterate over every `ready` grid in `VoxelManager` (same order as used when building descriptor sets). Track the smallest positive hit distance.
   - Apply a small safety margin (e.g. subtract 1–2 voxels) so zoom stops just before the surface.
   - If no grids report a hit, fall back to the existing fixed zoom distance.

5. **Feeding `InputController`**
   - Add a method (e.g. `App::nearest_voxel_along_camera(&Camera) -> Option<f64>`) that delegates to the `VoxelManager` raycast described above.
   - Call this helper from `InputController::update` (either by giving `InputController` a new trait object or by letting `App` perform the zoom translation itself after `input` reports scroll). The zoom delta becomes:

```rust
let base_step = scroll_delta * zoom_speed;
let limit = nearest_distance.unwrap_or(f64::INFINITY) - safety_margin;
let clamped_step = base_step.signum() * base_step.abs().min(limit.max(0.0));
camera.position += forward * clamped_step;
```

6. **Graceful fallbacks**
   - When no voxel caches exist yet (startup, grids still baking, cache evicted to save RAM), skip the raycast and reuse the constant step so UX stays responsive.
   - Expose a debug overlay / log message so we can confirm when zoom is surface-aware.

## Implementation Checklist
- [ ] Persist CPU copies of tile masks/payloads + derived dense bitsets (`GridOccupancyCache`).
- [ ] Cache `world_to_grid` matrices and voxel bounds per active grid.
- [ ] Implement a CPU ray/voxel traversal helper returning `Option<f64>` distance in world units.
- [ ] Expose the helper through `VoxelManager`/`VoxelSystem` to the main app loop.
- [ ] Use the returned distance to clamp camera translation inside `InputController` (or equivalent zoom handler).
- [ ] Add safety margin + fallback behavior.
- [ ] Optionally surface a debug UI readout to validate the computed distances.

This approach stays entirely on the CPU, reuses data we already generate, and avoids modifying the compute shader or render pipeline. It trades some additional memory for straightforward implementation and should be fast enough because DDA runs only when the player scrolls (a handful of voxels per event).
