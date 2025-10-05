use std::path::Path;
use std::sync::Arc;

use crate::hot_reload::HotReloadComputePipeline;
use vulkano::device::Device;

/// Centralized holder for all compute pipelines that can be hot reloaded.
pub struct PipelineManager {
    pub render: HotReloadComputePipeline,
    pub render_branchless: HotReloadComputePipeline,
    pub resample: HotReloadComputePipeline,
}

impl PipelineManager {
    pub fn new(device: Arc<Device>, shaders_dir: &Path) -> Self {
        let traverse_path = shaders_dir.join("traverse.comp");
        let resample_path = shaders_dir.join("resample.comp");
        let render = HotReloadComputePipeline::new(device.clone(), &traverse_path);
        let render_branchless = HotReloadComputePipeline::with_defines(
            device.clone(),
            &traverse_path,
            vec![("BRANCHLESS_TRAVERSAL".to_string(), None::<String>)],
        );
        let resample = HotReloadComputePipeline::new(device.clone(), &resample_path);
        Self { render, render_branchless, resample }
    }

    /// Reload shaders if the underlying source files changed.
    pub fn maybe_reload(&mut self) {
        self.render.maybe_reload();
        self.render_branchless.maybe_reload();
        self.resample.maybe_reload();
    }
}
