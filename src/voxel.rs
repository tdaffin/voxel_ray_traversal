use std::sync::Arc;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator};
use vulkano::pipeline::compute::ComputePipeline;
use vulkano::pipeline::Pipeline; // for layout() method
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator, MemoryTypeFilter};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, ClearColorImageInfo, CopyBufferToImageInfo, PrimaryCommandBufferAbstract};
use vulkano::device::Queue;

/// Build descriptor set (set=1) for an arbitrary number of voxel 3D image views.
/// The shader must declare matching bindings [0..N-1].
pub fn build_voxel_descriptor_set(
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    render_pipeline: &ComputePipeline,
    image_views: &[Arc<ImageView>],
) -> Arc<DescriptorSet> {
    assert!(!image_views.is_empty(), "Need at least one voxel image view");
    const MAX_GRIDS: usize = 32; // keep in sync with shader
    let layout = render_pipeline.layout().set_layouts()[1].clone();
    // Pad to MAX_GRIDS by repeating the first view. Shader only indexes [0, voxel_count),
    // so extra descriptors are never accessed; this avoids needing descriptor indexing features.
    let mut padded: Vec<Arc<ImageView>> = image_views.to_vec();
    while padded.len() < MAX_GRIDS { padded.push(padded[0].clone()); }
    let writes = [WriteDescriptorSet::image_view_array(0, 0, padded.iter().cloned())];
    DescriptorSet::new(
        descriptor_set_allocator,
        layout,
        writes,
        [],
    ).expect("Failed to create voxel descriptor set (array)")
}

/// Create a single 3D voxel image view and upload the provided packed voxel data.
pub fn create_voxel_image_view(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    voxels: Vec<u128>,
    resolution: u32,
) -> Arc<ImageView> {
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim3d,
            format: vulkano::format::Format::R32G32B32A32_UINT,
            extent: [resolution / 4, resolution / 4, resolution / 8],
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    ).expect("Failed to create voxel image");

    let src_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
        AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default() },
        voxels,
    ).expect("Failed to create staging buffer");

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).expect("Failed to create command buffer builder");
    command_buffer_builder
        .clear_color_image(ClearColorImageInfo::image(image.clone())).unwrap()
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(src_buffer, image.clone())).unwrap();
    let _ = command_buffer_builder.build().unwrap().execute(queue.clone()).unwrap();

    ImageView::new(image.clone(), vulkano::image::view::ImageViewCreateInfo::from_image(&image)).expect("Failed to create image view")
}

/// Batch-create voxel image views from multiple packed voxel arrays in parallel voxelization context.
pub fn create_voxel_image_views(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    all_voxels: Vec<Vec<u128>>,
    resolution: u32,
) -> Vec<Arc<ImageView>> {
    let mut views = Vec::with_capacity(all_voxels.len());
    for voxels in all_voxels.into_iter() {
        views.push(create_voxel_image_view(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
            voxels,
            resolution,
        ));
    }
    views
}
