use egui_winit_vulkano::egui::{self, Color32};

use crate::{app::App, model::Model, render_mode::RenderMode, rendering::get_images_and_sets};

/// Return values from a UI frame
/// (request_regen_voxels, trigger_benchmark)
pub(crate) type UiActions = (bool, bool);

impl App {
    /// Draws all egui windows and mutates self state accordingly.
    /// Returns (request_regen_voxels, trigger_benchmark)
    pub(crate) fn draw_ui(&mut self) -> UiActions {
        // Safe unwrap: render() only calls this after rcx creation in resumed()
        let rcx_for_ui = self.rcx.as_mut().unwrap();
        let mut trigger_benchmark = false;
        // Defer actions requiring &mut self after UI closure to avoid borrow conflicts.
        let mut request_regen_voxels = false;

        rcx_for_ui.gui.immediate_ui(|gui| {
            let ctx = gui.context();

            egui::Window::new("Settings").show(&ctx, |ui| {
                ui.style_mut().spacing.slider_width = 250.0;
                ui.horizontal(|ui| {
                    for &mode in RenderMode::ALL {
                        ui.selectable_value(&mut self.render_mode, mode, format!("{:?}", mode));
                    }
                });
                ui.separator();
                ui.add(
                    egui::Slider::new(
                        &mut self.voxel.manager.active_voxel_grids,
                        1..=Model::ALL.len() as u32,
                    )
                    .text("Active Grids"),
                );
                if ui.button("Benchmark Traversal Variants").clicked() {
                    trigger_benchmark = true;
                }
                ui.add(egui::Slider::new(&mut self.camera.fov, 0.0..=180.0).text("FOV"));
                // Render scale slider -> may recreate render & resample images
                if ui
                    .add(
                        egui::Slider::new(&mut self.render_scale, 0.125..=8.0).text("Render Scale"),
                    )
                    .changed()
                {
                    let window_extent: [u32; 2] = rcx_for_ui.window.inner_size().into();
                    let render_extent = [
                        (window_extent[0] as f32 * self.render_scale) as u32,
                        (window_extent[1] as f32 * self.render_scale) as u32,
                    ];
                    (
                        rcx_for_ui.render_image,
                        rcx_for_ui.render_set,
                        rcx_for_ui.resample_image,
                        rcx_for_ui.resample_set,
                    ) = get_images_and_sets(
                        self.gpu.memory_allocator.clone(),
                        self.gpu.descriptor_set_allocator.clone(),
                        &self.pipelines.render,
                        &self.pipelines.resample,
                        render_extent,
                        window_extent,
                    );
                }

                ui.colored_label(
                    Color32::LIGHT_RED,
                    "Warning: Very high resolutions may exhaust GPU memory.",
                );
                ui.label("Each grid can now have its own resolution (multiple of 8).");
                for i in 0..self.future_grid_resolutions.len() {
                    let mut val = self.future_grid_resolutions[i];
                    let label = format!("Grid {i} Res");
                    if ui.add(egui::Slider::new(&mut val, 8..=4096).text(label)).changed() {
                        // snap to multiple of 8
                        val = val.div_ceil(8) * 8;
                        self.future_grid_resolutions[i] = val;
                    }
                }
                ui.horizontal(|ui| {
                    for &model in Model::ALL {
                        ui.selectable_value(&mut self.model, model, format!("{:?}", model));
                    }
                });
                if ui.button("Regenerate Grids").clicked() {
                    request_regen_voxels = true;
                }
                if ui.button("Cancel Voxelization").clicked() {
                    self.voxel.manager.cancel();
                }
                ui.separator();
                // Progress overview
                let total = self.voxel.manager.voxel_pending.len();
                let remaining = self.voxel.manager.voxel_pending.iter().filter(|b| **b).count();
                ui.label(format!("Voxelization: {} / {} finished", total - remaining, total));
                for (i, (done, total_tris)) in self.voxel.manager.voxel_progress.iter().enumerate()
                {
                    let (d, t) = (*done, *total_tris);
                    let pct = if t > 0 { (d as f32 / t as f32 * 100.0).min(100.0) } else { 0.0 };
                    let status = if self.voxel.manager.voxel_pending[i] {
                        if t > 0 { format!("{pct:.1}%") } else { "…".into() }
                    } else {
                        "✓".into()
                    };
                    ui.label(format!("Grid {i}: {status}"));
                }
            });

            egui::Window::new("Stats").show(&ctx, |ui| {
                fn format_with_commas(n: u64) -> String {
                    let mut s = n.to_string();
                    let mut i = 3;
                    while i < s.len() {
                        s.insert(s.len() - i, ',');
                        i += 4;
                    }
                    s
                }

                ui.label(format!("FPS: {}", self.fps));

                let voxel_resolution = self.voxel.manager.voxel_resolution as u64;
                ui.label(format!(
                    "Voxels: {}³ = {}",
                    format_with_commas(voxel_resolution),
                    format_with_commas(voxel_resolution.pow(3))
                ));
                let window_extent: [u32; 2] = rcx_for_ui.window.inner_size().into();
                let render_extent = [
                    (window_extent[0] as f32 * self.render_scale) as u32,
                    (window_extent[1] as f32 * self.render_scale) as u32,
                ];
                ui.label(format!(
                    "Pixels: {}×{} = {}",
                    format_with_commas(window_extent[0] as u64),
                    format_with_commas(window_extent[1] as u64),
                    format_with_commas((window_extent[0] * window_extent[1]) as u64)
                ));
                ui.label(format!(
                    "Rays: {}×{} = {}",
                    format_with_commas(render_extent[0] as u64),
                    format_with_commas(render_extent[1] as u64),
                    format_with_commas((render_extent[0] * render_extent[1]) as u64)
                ));
            });
        });

        (request_regen_voxels, trigger_benchmark)
    }
}
