use egui_winit_vulkano::egui::{self, Color32};

use crate::{app::App, render_mode::RenderMode, rendering::get_images_and_sets};

/// Return values from a UI frame
/// (request_regen_voxels, trigger_benchmark)
pub(crate) type UiActions = (bool, bool);

impl App {
    /// Draws all egui windows and mutates self state accordingly.
    /// Returns (request_regen_voxels, trigger_benchmark)
    pub(crate) fn draw_ui(&mut self) -> UiActions {
        // Capture read-only stats before mutable borrows to avoid borrow conflicts
        let current_fps = self.fps();
        // Safe unwrap: render() only calls this after rcx creation in resumed()
        let rcx_for_ui = self.rcx.as_mut().unwrap();
        let mut trigger_benchmark = false;
        // Defer actions requiring &mut self after UI closure to avoid borrow conflicts.
        let mut request_regen_voxels = false;

        rcx_for_ui.gui.immediate_ui(|gui| {
            let ctx = gui.context();

            let translucent_fill = egui::Color32::from_rgba_unmultiplied(30, 30, 40, 120); // semi-transparent
            egui::Window::new("Settings")
                .frame(egui::Frame::window(&ctx.style()).fill(translucent_fill))
                .show(&ctx, |ui| {
                    ui.style_mut().spacing.slider_width = 250.0;
                    ui.horizontal(|ui| {
                        for &mode in RenderMode::ALL {
                            ui.selectable_value(&mut self.render_mode, mode, format!("{:?}", mode));
                        }
                    });
                    ui.separator();
                    let max_grids = self.voxel.manager.models.len().max(1) as u32;
                    ui.add(
                        egui::Slider::new(
                            &mut self.voxel.manager.active_voxel_grids,
                            1..=max_grids,
                        )
                        .text("Active Grids"),
                    );
                    if ui.button("Benchmark Traversal Variants").clicked() {
                        trigger_benchmark = true;
                    }
                    ui.add(egui::Slider::new(&mut self.camera.fov, 0.0..=180.0).text("FOV"));
                    ui.separator();
                    ui.label("Light Direction (spherical):");
                    static mut THETA: f32 = 0.9; // elevation
                    static mut PHI: f32 = 0.6; // azimuth
                    // Safe because single-threaded UI pass
                    let mut theta;
                    let mut phi;
                    unsafe {
                        theta = THETA;
                        phi = PHI;
                    }
                    if ui
                        .add(egui::Slider::new(&mut theta, -1.57..=1.57).text("Elevation"))
                        .changed()
                    {
                        unsafe {
                            THETA = theta;
                        }
                    }
                    if ui
                        .add(egui::Slider::new(&mut phi, -3.1415..=3.1415).text("Azimuth"))
                        .changed()
                    {
                        unsafe {
                            PHI = phi;
                        }
                    }
                    let ct = theta.cos();
                    self.light_dir = [ct * phi.cos(), theta.sin(), ct * phi.sin()];
                    // Render scale slider -> may recreate render & resample images
                    if ui
                        .add(
                            egui::Slider::new(&mut self.render_scale, 0.125..=8.0)
                                .text("Render Scale"),
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

                    // Advanced section collapsed by default
                    egui::CollapsingHeader::new("Advanced").default_open(false).show(ui, |ui| {
                        ui.colored_label(
                            Color32::LIGHT_RED,
                            "Warning: Very high resolutions may exhaust GPU memory.",
                        );
                        ui.label(
                            "Each (resizable) grid can have its own resolution (multiple of 8).",
                        );
                        for i in 0..self.future_grid_resolutions.len() {
                            // Hide resolution slider for native .vox models which use intrinsic dimensions
                            let is_vox = self
                                .voxel
                                .manager
                                .models
                                .get(i)
                                .map(|m| m.extension.as_str() == "vox")
                                .unwrap_or(false);
                            if is_vox {
                                ui.label(format!("Grid {i} Res: native (.vox)"));
                                continue;
                            }
                            let mut val = self.future_grid_resolutions[i];
                            let label = format!("Grid {i} Res");
                            if ui.add(egui::Slider::new(&mut val, 8..=4096).text(label)).changed() {
                                // snap to multiple of 8
                                val = val.div_ceil(8) * 8;
                                self.future_grid_resolutions[i] = val;
                            }
                        }
                        ui.label("Discovered Models:");
                        for (i, m) in self.voxel.manager.models.iter().enumerate() {
                            // Fetch actual dimensions if available
                            let dims = self.voxel.manager.grid_dims.get(i).copied().unwrap_or((
                                self.voxel.manager.grid_resolutions.get(i).copied().unwrap_or(0),
                                0,
                                0,
                            ));
                            let (dx, dy, dz) = dims;
                            let storage =
                                self.voxel.manager.grid_resolutions.get(i).copied().unwrap_or(0);
                            let dim_str = if dx == dy && dy == dz {
                                format!("{}³", dx)
                            } else {
                                format!("{}×{}×{}", dx, dy, dz)
                            };
                            let storage_note = if storage != dx || storage != dy || storage != dz {
                                format!(" (storage cube {}³)", storage)
                            } else {
                                String::new()
                            };
                            ui.label(format!(
                                "{}: {} (.{}) dims={}{}",
                                i, m.name, m.extension, dim_str, storage_note
                            ));
                        }
                        if ui.button("Regenerate Grids").clicked() {
                            request_regen_voxels = true;
                        }
                        if ui.button("Cancel Voxelization").clicked() {
                            self.voxel.manager.cancel();
                        }
                        ui.separator();
                        // Progress overview
                        let total = self.voxel.manager.voxel_pending.len();
                        let remaining =
                            self.voxel.manager.voxel_pending.iter().filter(|b| **b).count();
                        ui.label(format!(
                            "Voxelization: {} / {} finished",
                            total - remaining,
                            total
                        ));
                        for (i, (done, total_tris)) in
                            self.voxel.manager.voxel_progress.iter().enumerate()
                        {
                            let (d, t) = (*done, *total_tris);
                            let pct =
                                if t > 0 { (d as f32 / t as f32 * 100.0).min(100.0) } else { 0.0 };
                            let status = if self.voxel.manager.voxel_pending[i] {
                                if t > 0 { format!("{pct:.1}%") } else { "…".into() }
                            } else {
                                "✓".into()
                            };
                            ui.label(format!("Grid {i}: {status}"));
                        }
                    });
                });

            let fps_val = current_fps; // captured outside mutable borrow of self
            egui::Window::new("Stats")
                .frame(egui::Frame::window(&ctx.style()).fill(translucent_fill))
                .show(&ctx, |ui| {
                    fn format_with_commas(n: u64) -> String {
                        let mut s = n.to_string();
                        let mut i = 3;
                        while i < s.len() {
                            s.insert(s.len() - i, ',');
                            i += 4;
                        }
                        s
                    }

                    ui.label(format!("FPS: {}", fps_val));

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
                    // List active grid dims summary
                    if !self.voxel.manager.grid_dims.is_empty() {
                        ui.separator();
                        ui.label("Grid Dimensions:");
                        for (i, (dx, dy, dz)) in self.voxel.manager.grid_dims.iter().enumerate() {
                            ui.label(format!("Grid {i}: {}×{}×{}", dx, dy, dz));
                        }
                    }
                });
        });

        (request_regen_voxels, trigger_benchmark)
    }
}
