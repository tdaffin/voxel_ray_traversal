use std::sync::Arc;

use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator, device::Queue,
    memory::allocator::StandardMemoryAllocator,
};

use crate::{
    hot_reload::HotReloadComputePipeline, model_discovery::DiscoveredModel, voxel_job::VoxelManager,
};

/// Thin facade that captures the frequently repeated allocator & queue Arcs
/// so call sites only provide what semantically changes (e.g. target pipeline
/// or placeholder resolution). This reduces parameter noise in `App`.
pub struct VoxelSystem {
    pub manager: VoxelManager,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
}

impl VoxelSystem {
    pub fn new(
        initial_resolution: u32, memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>, queue: Arc<Queue>,
        render_pipeline: &HotReloadComputePipeline, models: Option<Vec<DiscoveredModel>>,
    ) -> Self {
        let manager = VoxelManager::new(
            initial_resolution,
            memory_allocator.clone(),
            descriptor_set_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            render_pipeline,
            models,
        );
        Self {
            manager,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            queue,
        }
    }

    pub fn poll(&mut self, render_pipeline: &HotReloadComputePipeline, delta_seconds: f32) {
        self.manager.poll(
            self.descriptor_set_allocator.clone(),
            render_pipeline,
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.queue.clone(),
            delta_seconds,
        );
    }

    pub fn regenerate(
        &mut self, render_pipeline: &HotReloadComputePipeline, placeholder_resolution: u32,
    ) {
        self.manager.regenerate(
            self.descriptor_set_allocator.clone(),
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.queue.clone(),
            render_pipeline,
            placeholder_resolution,
        );
    }

    pub fn regenerate_grid(&mut self, index: usize, render_pipeline: &HotReloadComputePipeline) {
        self.manager.regenerate_grid(
            index,
            self.descriptor_set_allocator.clone(),
            self.memory_allocator.clone(),
            render_pipeline,
        );
    }

    pub fn cancel_grid(&mut self, index: usize) {
        self.manager.cancel_grid(index);
    }

    pub fn add_models(
        &mut self, models: Vec<DiscoveredModel>, render_pipeline: &HotReloadComputePipeline,
        placeholder_resolution: u32,
    ) {
        self.manager.add_models(
            models,
            self.descriptor_set_allocator.clone(),
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.queue.clone(),
            render_pipeline,
            placeholder_resolution,
        );
    }

    pub fn set_models(
        &mut self, models: Vec<DiscoveredModel>, render_pipeline: &HotReloadComputePipeline,
        placeholder_resolution: u32,
    ) {
        self.manager.set_models(
            models,
            self.descriptor_set_allocator.clone(),
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.queue.clone(),
            render_pipeline,
            placeholder_resolution,
        );
    }

    pub fn randomize_rotation(&mut self, index: usize, render_pipeline: &HotReloadComputePipeline) {
        self.manager.apply_random_rotation(
            index,
            self.descriptor_set_allocator.clone(),
            render_pipeline,
            self.memory_allocator.clone(),
        );
    }
}

impl std::ops::Deref for VoxelSystem {
    type Target = VoxelManager;
    fn deref(&self) -> &Self::Target {
        &self.manager
    }
}

impl std::ops::DerefMut for VoxelSystem {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.manager
    }
}

// Legacy cancel forwarder removed; call self.manager.cancel() at call sites instead.
