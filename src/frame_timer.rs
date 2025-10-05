use std::time::{Duration, Instant};

/// Tracks frame count and computes Frames Per Second (FPS) once per second.
///
/// Usage:
/// ```ignore
/// let mut timer = FrameTimer::new();
/// // each frame:
/// if let Some(fps) = timer.frame() {
///     println!("fps: {fps}");
/// }
/// ```
#[derive(Debug, Clone)]
pub struct FrameTimer {
    last_second: Instant,
    frames_since_last_second: u32,
    fps: u32,
}

impl FrameTimer {
    /// Create a new frame timer starting now.
    pub fn new() -> Self {
        Self { last_second: Instant::now(), frames_since_last_second: 0, fps: 0 }
    }

    /// Notify the timer that one frame has completed. Returns `Some(fps)`
    /// when one second has elapsed and the FPS value has updated, otherwise `None`.
    pub fn frame(&mut self) -> Option<u32> {
        self.frames_since_last_second += 1;
        let now = Instant::now();
        if now.duration_since(self.last_second) >= Duration::from_secs(1) {
            self.fps = self.frames_since_last_second;
            self.frames_since_last_second = 0;
            self.last_second = now;
            return Some(self.fps);
        }
        None
    }

    /// Current FPS value (last computed). Will be 0 until at least one full second passes.
    pub fn fps(&self) -> u32 {
        self.fps
    }
}
