use egui_winit_vulkano::egui::{self, Color32};

use crate::{app::App, render_mode::RenderMode};

/// Return values from a UI frame
/// (request_regen_voxels, trigger_benchmark)
pub(crate) type UiActions = (bool, bool);

fn format_with_commas(n: u64) -> String {
    let mut s = n.to_string();
    let mut i = 3;
    while i < s.len() {
        s.insert(s.len() - i, ',');
        i += 4;
    }
    s
}

impl App {
    /// Draws all egui windows and mutates self state accordingly.
    /// Returns (request_regen_voxels, trigger_benchmark)
    pub(crate) fn draw_ui(&mut self) -> UiActions {
        // Capture read-only stats before mutable borrows to avoid borrow conflicts
        let current_fps = self.fps();
        // Safe unwrap: render() only calls this after rcx creation in resumed()
        let mut trigger_benchmark = false;
        // Defer actions requiring &mut self after UI closure to avoid borrow conflicts.
        let mut request_regen_voxels = false;

        let mut rebuild_render_targets = false;
        let mut mark_state_dirty = false;
        {
            let rcx_for_ui = self.rcx.as_mut().unwrap();
            rcx_for_ui.gui.immediate_ui(|gui| {
            let ctx = gui.context();

            let translucent_fill = egui::Color32::from_rgba_unmultiplied(30, 30, 40, 120); // semi-transparent
            egui::Window::new("Settings")
                .frame(egui::Frame::window(&ctx.style()).fill(translucent_fill))
                .show(&ctx, |ui| {
                    let mut ui_state_changed = false;
                    ui.style_mut().spacing.slider_width = 250.0;
                    ui.horizontal(|ui| {
                        let mut mode_changed = false;
                        for &mode in RenderMode::ALL {
                            let response =
                                ui.selectable_value(&mut self.render_mode, mode, format!("{:?}", mode));
                            if response.changed() {
                                mode_changed = true;
                            }
                        }
                        if mode_changed {
                            ui_state_changed = true;
                        }
                    });
                    if ui
                        .checkbox(&mut self.always_instant, "Full grids")
                        .on_hover_text(
                            "When enabled, grids register as a hit immediately upon entry.\nDisable to require sampling a filled bit first."
                        )
                        .changed()
                    {
                        ui_state_changed = true;
                    }
                    if ui
                        .checkbox(&mut self.hit_back, "Hit grid backs")
                        .on_hover_text(
                            "When enabled, rays register a hit when they exit a grid without finding a filled voxel."
                        )
                        .changed()
                    {
                        ui_state_changed = true;
                    }
                    if ui
                        .checkbox(&mut self.verbose_logging, "Verbose logging")
                        .on_hover_text("Print detailed palette and compression stats to stdout")
                        .changed()
                    {
                        crate::log_config::set_verbose_logging(self.verbose_logging);
                        ui_state_changed = true;
                    }
                    ui.separator();
                    let max_grids = (self.voxel.manager.models.len() + 1).max(1) as u32;
                    let active_slider = egui::Slider::new(
                        &mut self.voxel.manager.active_voxel_grids,
                        1..=max_grids,
                    )
                    .text("Active Grids (incl. ground)");
                    if ui.add(active_slider).changed() {
                        ui_state_changed = true;
                    }
                    if ui.button("Benchmark Traversal Variants").clicked() {
                        trigger_benchmark = true;
                    }
                    if ui
                        .add(egui::Slider::new(&mut self.camera.fov, 0.0..=180.0).text("FOV"))
                        .changed()
                    {
                        ui_state_changed = true;
                    }
                    let mut mouse_sense = self.camera.mouse_sensitivity as f32;
                    if ui
                        .add(
                            egui::Slider::new(&mut mouse_sense, 0.0005..=0.01)
                                .text("Mouse Sensitivity"),
                        )
                        .changed()
                    {
                        self.camera.mouse_sensitivity = mouse_sense as f64;
                        ui_state_changed = true;
                    }
                    ui.separator();
                    ui.label("Light Direction (spherical):");
                    let mut theta = self.light_theta;
                    let mut phi = self.light_phi;
                    if ui
                        .add(egui::Slider::new(&mut theta, -1.57..=1.57).text("Elevation"))
                        .changed()
                    {
                        self.light_theta = theta;
                        let theta_val = self.light_theta;
                        let phi_val = self.light_phi;
                        let ct = theta_val.cos();
                        self.light_dir = [ct * phi_val.cos(), theta_val.sin(), ct * phi_val.sin()];
                        ui_state_changed = true;
                    }
                    if ui
                        .add(egui::Slider::new(&mut phi, -3.1415..=3.1415).text("Azimuth"))
                        .changed()
                    {
                        self.light_phi = phi;
                        let theta_val = self.light_theta;
                        let phi_val = self.light_phi;
                        let ct = theta_val.cos();
                        self.light_dir = [ct * phi_val.cos(), theta_val.sin(), ct * phi_val.sin()];
                        ui_state_changed = true;
                    }
                    // Render scale slider -> may recreate render & resample images
                    let render_scale_response = ui.add(
                        egui::Slider::new(&mut self.render_scale, 0.125..=8.0)
                            .text("Render Scale"),
                    );
                    if render_scale_response.changed() {
                        rebuild_render_targets = true;
                        ui_state_changed = true;
                    }

                    ui.separator();
                    if ui
                        .button("Advanced Tools…")
                        .on_hover_text("Open detailed voxel and model controls")
                        .clicked()
                    {
                        self.advanced_window_open = true;
                        ui_state_changed = true;
                    }

                    if ui.button("Reset UI to Defaults").clicked() {
                        let defaults = crate::ui_state::PersistedUiState::default();
                        let previous_scale = self.render_scale;
                        self.render_mode = defaults.render_mode;
                        self.always_instant = defaults.always_instant;
                        self.hit_back = defaults.hit_back;
                        self.verbose_logging = defaults.verbose_logging;
                        crate::log_config::set_verbose_logging(self.verbose_logging);
                        self.render_scale = defaults.render_scale;
                        let max_grids = (self.voxel.manager.models.len() + 1).max(1) as u32;
                        self.voxel.manager.active_voxel_grids =
                            defaults.active_voxel_grids.clamp(1, max_grids);
                        self.camera.fov = defaults.camera_fov;
                        self.camera.mouse_sensitivity = defaults.mouse_sensitivity;
                        self.light_theta = defaults.light_theta;
                        self.light_phi = defaults.light_phi;
                        let theta_val = self.light_theta;
                        let phi_val = self.light_phi;
                        let ct = theta_val.cos();
                        self.light_dir = [ct * phi_val.cos(), theta_val.sin(), ct * phi_val.sin()];
                        self.advanced_window_open = defaults.advanced_window_open;
                        self.future_grid_resolutions =
                            self.voxel.manager.grid_resolutions.clone();
                        self.voxel.manager.future_grid_resolutions =
                            self.future_grid_resolutions.clone();
                        if (self.render_scale - previous_scale).abs() > f32::EPSILON {
                            rebuild_render_targets = true;
                        }
                        ui_state_changed = true;
                    }

                    if ui_state_changed {
                        mark_state_dirty = true;
                    }
                });

            let fps_val = current_fps; // captured outside mutable borrow of self
            egui::Window::new("Stats")
                .frame(egui::Frame::window(&ctx.style()).fill(translucent_fill))
                .show(&ctx, |ui| {
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
                });

            let advanced_open_before = self.advanced_window_open;
            egui::Window::new("Advanced Tools")
                .open(&mut self.advanced_window_open)
                .frame(egui::Frame::window(&ctx.style()).fill(translucent_fill))
                .show(&ctx, |ui| {
                    ui.colored_label(
                        Color32::LIGHT_RED,
                        "Warning: Very high resolutions may exhaust GPU memory.",
                    );
                    ui.label("Each (resizable) grid can have its own resolution (multiple of 8).");
                    ui.separator();

                    for i in 0..self.future_grid_resolutions.len() {
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
                            val = val.div_ceil(8) * 8;
                            self.future_grid_resolutions[i] = val;
                            if let Some(entry) =
                                self.voxel.manager.future_grid_resolutions.get_mut(i)
                            {
                                *entry = val;
                            }
                            mark_state_dirty = true;
                        }
                    }

                    ui.separator();
                    ui.label("Discovered Models:");
                    let model_count = self.voxel.manager.models.len();
                    for i in 0..model_count {
                        let model = &self.voxel.manager.models[i];
                        let dims = self.voxel.manager.grid_dims.get(i).copied().unwrap_or((
                            self.voxel.manager.grid_resolutions.get(i).copied().unwrap_or(0),
                            0,
                            0,
                        ));
                        let storage_dims = self
                            .voxel
                            .manager
                            .grid_storage
                            .get(i)
                            .copied()
                            .unwrap_or((0, 0, 0));
                        let (dx, dy, dz) = dims;
                        let dim_str = if dx == dy && dy == dz {
                            format!("{}³", dx)
                        } else {
                            format!("{}×{}×{}", dx, dy, dz)
                        };
                        let storage_note = if storage_dims != (dx, dy, dz) {
                            format!(
                                " (storage {}×{}×{})",
                                storage_dims.0, storage_dims.1, storage_dims.2
                            )
                        } else {
                            String::new()
                        };
                        let label_text = format!(
                            "{}: {} (.{} ) dims={}{}",
                            i, model.name, model.extension, dim_str, storage_note
                        );
                        let speed_rad = self
                            .voxel
                            .manager
                            .grid_rotation_speeds
                            .get(i)
                            .copied()
                            .unwrap_or(0.0);
                        let mut speed_deg = speed_rad.to_degrees();
                        let mut speed_changed = false;
                        ui.horizontal(|ui| {
                            ui.label(&label_text);
                            if ui
                                .button(format!("Random rotation##{}", i))
                                .on_hover_text("Apply a fresh random orientation to this grid")
                                .clicked()
                            {
                                self.voxel.randomize_rotation(i, &self.pipelines.render);
                            }
                            ui.label("Spin:");
                            let response = ui
                                .add(
                                    egui::DragValue::new(&mut speed_deg).speed(5.0).suffix("°/s"),
                                )
                                .on_hover_text("Rotation speed around the grid's vertical axis");
                            if response.changed() {
                                speed_changed = true;
                            }
                        });
                        if speed_changed {
                            if let Some(entry) = self.voxel.manager.grid_rotation_speeds.get_mut(i)
                            {
                                *entry = speed_deg.to_radians();
                            }
                        }

                        if self
                            .voxel
                            .manager
                            .voxel_pending
                            .get(i)
                            .copied()
                            .unwrap_or(true)
                        {
                            ui.label("Compression: pending…");
                        } else if let Some(stats) = self
                            .voxel
                            .manager
                            .tile_stats
                            .get(i)
                            .and_then(|s| s.as_ref())
                        {
                            ui.label(format!(
                                "Compression: empty={} uniform={} dense={}",
                                format_with_commas(stats.empty_tiles as u64),
                                format_with_commas(stats.uniform_tiles as u64),
                                format_with_commas(stats.dense_tiles as u64)
                            ));
                        }
                    }

                    ui.separator();
                    if ui.button("Regenerate Grids").clicked() {
                        request_regen_voxels = true;
                    }
                    if ui.button("Cancel Voxelization").clicked() {
                        self.voxel.manager.cancel();
                    }

                    ui.separator();
                    let total = self.voxel.manager.voxel_pending.len();
                    let remaining = self.voxel.manager.voxel_pending.iter().filter(|b| **b).count();
                    ui.label(format!(
                        "Voxelization: {} / {} finished",
                        total - remaining,
                        total
                    ));
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
            if self.advanced_window_open != advanced_open_before {
                mark_state_dirty = true;
            }
        });
        }

        if rebuild_render_targets {
            self.apply_render_scale_change();
        }
        if mark_state_dirty {
            self.mark_ui_state_dirty();
        }
        self.flush_ui_state();

        (request_regen_voxels, trigger_benchmark)
    }
}
