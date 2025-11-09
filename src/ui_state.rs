use std::path::{Path, PathBuf};

use dirs::config_dir;
use serde::{Deserialize, Serialize};

use crate::render_mode::RenderMode;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistedUiState {
    pub render_mode: RenderMode,
    pub always_instant: bool,
    pub hit_back: bool,
    pub verbose_logging: bool,
    pub render_scale: f32,
    pub active_voxel_grids: u32,
    pub camera_fov: f64,
    pub mouse_sensitivity: f64,
    pub light_theta: f32,
    pub light_phi: f32,
    pub advanced_window_open: bool,
    pub future_grid_resolutions: Vec<u32>,
    pub loaded_models: Vec<String>,
}

impl Default for PersistedUiState {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::Shade,
            always_instant: false,
            hit_back: false,
            verbose_logging: false,
            render_scale: 0.25,
            active_voxel_grids: 1,
            camera_fov: 35.0,
            mouse_sensitivity: crate::camera::Camera::DEFAULT_MOUSE_SENSITIVITY,
            light_theta: 0.9,
            light_phi: 0.6,
            advanced_window_open: false,
            future_grid_resolutions: Vec::new(),
            loaded_models: Vec::new(),
        }
    }
}

pub fn default_ui_state_path() -> Option<PathBuf> {
    config_dir().map(|dir| dir.join("voxel_ray_traversal").join("ui_state.json"))
}

pub fn load_ui_state(path: &Path) -> Option<PersistedUiState> {
    std::fs::read_to_string(path).ok().and_then(|contents| serde_json::from_str(&contents).ok())
}

pub fn save_ui_state(path: &Path, state: &PersistedUiState) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(state)?;
    std::fs::write(path, json)
}
