# Adaptive Depth Sampling for Camera Zoom

This note builds on `docs/camera_zoom_depth_buffer.md` and focuses on obtaining a reliable depth estimate when the center pixel is empty (e.g., looking into the sky). The goal is to keep per-frame costs low by sampling a tiny cluster of pixels **only when the user scroll-zooms**.

## Overview
1. Render pass already writes world-space hit distances into a `R32_SFLOAT` depth image (as described in the depth-buffer doc).
2. Instead of always reading one center pixel, we sample a small pattern on-demand:
   - Query a list of candidate offsets around the center.
   - Read pixels sequentially until we find a valid (non-sentinel) depth.
   - Cache the result for this zoom gesture.
3. If all candidates miss (scene truly empty), fall back to the legacy fixed zoom distance.

## Sampling Pattern
- Use a spiral or concentric ring of offsets to cover nearby pixels without branching widely.
- Example offsets ordered by priority (in render-resolution texels):
```
[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
 (1, 1), (-1, 1), (1, -1), (-1, -1),
 (2, 0), (-2, 0), (0, 2), (0, -2), ...]
```
- Limit the search radius (e.g., up to 16 pixels) to cap worst-case readbacks.
- Because the render image is supersampled relative to the swapchain, these offsets cover several arc-minutes of FOV and usually catch geometry.

## GPU Readback Strategy
- Store candidate offsets in CPU memory; convert them to absolute coordinates each time you need a measurement.
- On zoom input:
  1. Build an inline command buffer (or reuse the main frame buffer) that issues **K** `vkCmdCopyImageToBuffer` calls, each copying a 1×1 block from `depth_image` into a contiguous staging buffer of size `K * 4` bytes.
  2. Submit once and wait (or piggyback on the render future if zoom happens immediately after a frame was drawn).
  3. Map the staging buffer and scan for the first finite depth value.
- Keep `K` small (e.g., 9–16). For a 16-sample search, the GPU copies only 16 pixels per zoom event—negligible compared to dispatch time.

## Event-Driven Optimization
- Only perform the multi-sample readback when scroll input occurs and the last cached depth is either missing or older than a frame.
- If the user scrolls repeatedly within a short window, reuse the last good depth until the camera moves significantly (e.g., track a small epsilon displacement threshold before invalidating the cached value).

## Combining with Safety Margins
- Once a depth value is found, apply the same clamping logic from the base depth-buffer approach: `allowed_zoom = min(requested_zoom, depth - margin)`.
- Optionally bias the selected depth slightly toward the center by adding a penalty proportional to the offset magnitude; this prevents distant peripheral hits from halting zoom prematurely.

## Failure Modes and Fallbacks
- If all sampled pixels return `FLT_MAX`, treat the result as “no hit” and fall back to constant zoom.
- Ensure the command buffer is skipped entirely when the renderer hasn’t produced a depth image yet (startup, swapchain recreation, etc.).
- For future features (focus reticle), simply shift the center of the offset pattern to the reticle location.

## Implementation Checklist
- [ ] Define a compile-time or runtime list of pixel offsets describing the search pattern.
- [ ] Allocate a staging buffer large enough for `max_samples * 4` bytes.
- [ ] On zoom, record image-to-buffer copies for each offset until a hit is found (allow early exit once a valid depth is read).
- [ ] Cache and reuse the hit value for subsequent scroll ticks during the same gesture.
- [ ] Add heuristics to invalidate the cache when the camera moves sideways or when a new frame renders (to avoid staleness).

This adaptive sampling keeps per-frame rendering untouched while ensuring zoom has a reasonable depth target even when the center pixel faces empty space.
