use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum RenderMode {
    Coord = 0,
    Steps = 1,
    Normal = 2,
    UV = 3,
    Depth = 4,
    Shade = 5, // New shaded lighting + AO mode (matches SHADE in traverse.comp)
    Debug = 6,
}

impl RenderMode {
    pub const ALL: &'static [RenderMode] = &[
        RenderMode::Coord,
        RenderMode::Steps,
        RenderMode::Normal,
        RenderMode::UV,
        RenderMode::Depth,
        RenderMode::Shade,
        RenderMode::Debug,
    ];
}
