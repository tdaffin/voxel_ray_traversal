#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Coord,
    Steps,
    Normal,
    UV,
    Depth,
}

impl RenderMode {
    pub const ALL: &'static [RenderMode] = &[
        RenderMode::Coord,
        RenderMode::Steps,
        RenderMode::Normal,
        RenderMode::UV,
        RenderMode::Depth,
    ];
}
