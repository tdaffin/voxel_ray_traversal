use std::sync::Arc;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator};
use vulkano::pipeline::compute::ComputePipeline;
use vulkano::pipeline::Pipeline; // for layout() method
use vulkano::image::view::ImageView;

/// Build descriptor set (set=1) for an arbitrary number of voxel 3D image views.
/// The shader must declare matching bindings [0..N-1].
pub fn build_voxel_descriptor_set(
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    render_pipeline: &ComputePipeline,
    image_views: &[Arc<ImageView>],
) -> Arc<DescriptorSet> {
    assert!(!image_views.is_empty(), "Need at least one voxel image view");
    let layout = render_pipeline.layout().set_layouts()[1].clone();
    let writes: Vec<WriteDescriptorSet> = image_views
        .iter()
        .enumerate()
        .map(|(i, view)| WriteDescriptorSet::image_view(i as u32, view.clone()))
        .collect();
    DescriptorSet::new(
        descriptor_set_allocator,
        layout,
        writes,
        [],
    ).expect("Failed to create voxel descriptor set")
}
