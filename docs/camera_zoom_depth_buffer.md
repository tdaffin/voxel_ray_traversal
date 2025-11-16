# Depth-Buffer-Assisted Camera Zoom

Goal: cap scroll zoom based on the depth of the nearest surface already computed by the renderer, avoiding extra CPU-side voxel traversal.

This approach reuses the compute shader’s existing hit-distance calculations and samples a previously rendered depth value (e.g., the center pixel) to determine how far the camera should move.

## Existing Rendering Flow Recap
- `frame_renderer::record_frame` dispatches the `traverse.comp` compute shader into `rcx.render_image` (RGBA8 storage image), then resamples/blits it to the swapchain.
- The shader already computes `HitAttributes.hitDistance` (world-space length from ray origin to hit point) for every shaded pixel but currently discards it after converting to color.
- No depth information is persisted per frame.

## Minimal Depth Buffer Extension
1. **Add a storage image for depth**
   - Extend `RenderContext` (in `rendering.rs`) to allocate an additional single-channel image, e.g. `Format::R32_SFLOAT`, matching the render extent.
   - Bind it as a second descriptor in set 0 (e.g. `layout(set = 0, binding = 1, r32f) writeonly uniform image2D depthImage;`).
   - Update descriptor-set creation (`get_images_and_sets`) and `frame_renderer::record_frame` to include this image when binding the render pipeline.

2. **Write depth in the compute shader**
   - Inside `traverse.comp`, after computing the final color, store the hit distance: `imageStore(depthImage, ivec2(gl_GlobalInvocationID.xy), vec4(hit.hitDistance, 0, 0, 0));`.
   - For rays that miss, write a sentinel (e.g. `FLT_MAX`).
   - Optionally clamp / encode logarithmically if needed, but raw meters are fine for zooming.

3. **Copy one pixel back to the CPU**
   - Allocate a tiny persistent staging buffer (16 bytes) and a matching `GpuFuture` fence.
   - After the render dispatch (before presenting), enqueue `copy_image_to_buffer` to copy a 1×1 region from the depth image at the desired screen coordinate (typically the window center) into the buffer.
   - Submit + wait via the same future chain already used for GUI rendering (the copy becomes part of the command buffer built in `record_frame`).
   - Map the staging buffer once per frame (or only on demand) to read the float depth.

4. **Feed Input Controller**
   - Store the latest depth sample in `App` (e.g. `self.last_depth_sample: Option<f32>`), updating it when the readback completes.
   - When handling scroll input, compute the camera’s forward translation as `min(requested_zoom, depth_sample - safety_margin)` if the depth sample is valid (< `FLT_MAX`).
   - A small safety margin (e.g. `0.05` world units) prevents clipping into the surface.

5. **Handling dynamic resolution / reticle**
   - Depth sampling should use the same pixel coordinates as the user’s “aim” point. For a center-locked camera, this is `(render_width/2, render_height/2)` regardless of window resample scale.
   - If you later add a mouse-driven focus point, simply convert the cursor position to render-resolution coordinates and copy that pixel instead.

6. **Fallback behavior**
   - If the copy hasn’t finished (e.g., first frame, or frame skipped), treat the depth value as `None` and fall back to the old fixed zoom distance.
   - Consider debouncing readbacks so zooming doesn’t wait for multiple frames; even copying once per frame is cheap because it’s a 1×1 region.

## Advantages & Tradeoffs
- **Pros**: zero CPU duplication of the voxel traversal/lighting math; always matches the exact shader output; trivial to extend to other features (focus-peaking, DOF, etc.).
- **Cons**: requires an extra storage image binding and a 1×1 readback each frame (but cost is minimal). If the center pixel looks at empty space, zoom still falls back to constant speed (works out-of-the-box thanks to the sentinel value).

## Implementation Checklist
- [ ] Allocate `depth_image` + descriptor in `RenderContext` and pass it to the render pipeline bindings.
- [ ] Update `traverse.comp` to declare the new image and store hit/miss distances per pixel.
- [ ] Extend `frame_renderer::record_frame` to copy a chosen pixel from `depth_image` into a host-visible staging buffer each frame.
- [ ] Track the latest depth sample in `App` and expose it to the zoom handler.
- [ ] Clamp scroll-based camera translations to the sampled depth minus a safety offset.
- [ ] Preserve fallbacks when no valid depth is available.

This document complements `docs/camera_zoom_distance.md`: both paths rely on existing data, but the depth-buffer approach avoids CPU-side voxel math at the cost of a tiny GPU readback.
