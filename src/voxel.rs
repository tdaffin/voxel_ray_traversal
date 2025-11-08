use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyBufferToImageInfo,
    PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{
    DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
};
use vulkano::device::Queue;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::Pipeline; // for layout() method
use vulkano::pipeline::compute::ComputePipeline;

/// Descriptor set index for voxel resources in the traversal pipeline.
#[allow(dead_code)]
pub const VOXEL_DESCRIPTOR_SET_INDEX: u32 = 1;
pub const BINDING_VOXEL_IMAGES: u32 = 0;
pub const BINDING_COLOR_INDICES: u32 = 1;
pub const BINDING_PALETTE_BUFFER: u32 = 2;
pub const BINDING_GRID_INFO: u32 = 3;
/// Reserved binding numbers for upcoming tile mask + payload resources.
#[allow(dead_code)]
pub const BINDING_TILE_MASK: u32 = 4;
#[allow(dead_code)]
pub const BINDING_TILE_PAYLOADS: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, BufferContents)]
pub struct GridInfo {
    pub resolution: u32, // cubic storage resolution (power-of-two-ish padded)
    pub dim_x: u32,      // actual content dimensions (<= resolution)
    pub dim_y: u32,
    pub dim_z: u32,
    // New: storage extents (packed texture actual allocated dimensions). For now mirror cubic; will diverge when non-cubic storage enabled.
    pub storage_w: u32,    // packed voxel texel width (resolution/4 currently)
    pub storage_h: u32,    // packed voxel texel height (resolution/4 currently)
    pub storage_d: u32,    // packed voxel texel depth (resolution/8 currently)
    pub palette_base: u32, // starting index into global palette buffer
    pub palette_len: u32,  // number of valid palette entries for this grid
    pub _padding0: u32,
    pub _padding1: u32,
    pub _padding2: u32,
    pub grid_to_world: [[f32; 4]; 4],
}

impl Default for GridInfo {
    fn default() -> Self {
        Self {
            resolution: 0,
            dim_x: 0,
            dim_y: 0,
            dim_z: 0,
            storage_w: 0,
            storage_h: 0,
            storage_d: 0,
            palette_base: 0,
            palette_len: 0,
            _padding0: 0,
            _padding1: 0,
            _padding2: 0,
            grid_to_world: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

pub fn create_grid_info_buffer(
    memory_allocator: Arc<StandardMemoryAllocator>, infos: &[GridInfo],
) -> Subbuffer<[GridInfo]> {
    let usage = BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST;
    Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo { usage, ..Default::default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        infos.iter().cloned(),
    )
    .expect("Failed to create grid info buffer")
}

/// Build descriptor set (set=1) for an arbitrary number of voxel 3D image views.
/// The shader must declare matching bindings [0..N-1].
pub fn build_voxel_descriptor_set(
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    render_pipeline: &ComputePipeline, image_views: &[Arc<ImageView>],
    color_index_views: &[Arc<ImageView>], palette_buffer: Subbuffer<[[f32; 4]]>,
    grid_info_buffer: Subbuffer<[GridInfo]>,
) -> Arc<DescriptorSet> {
    assert!(!image_views.is_empty(), "Need at least one voxel image view");
    const MAX_GRIDS: usize = 32; // keep in sync with shader
    let layout = render_pipeline.layout().set_layouts()[1].clone();
    // Pad to MAX_GRIDS by repeating the first view. Shader only indexes [0, voxel_count),
    // so extra descriptors are never accessed; this avoids needing descriptor indexing features.
    let mut padded: Vec<Arc<ImageView>> = image_views.to_vec();
    while padded.len() < MAX_GRIDS {
        padded.push(padded[0].clone());
    }
    let mut padded_colors: Vec<Arc<ImageView>> = color_index_views.to_vec();
    while padded_colors.len() < MAX_GRIDS {
        padded_colors.push(padded_colors[0].clone());
    }
    let writes = [
        WriteDescriptorSet::image_view_array(BINDING_VOXEL_IMAGES, 0, padded.iter().cloned()),
        WriteDescriptorSet::image_view_array(
            BINDING_COLOR_INDICES,
            0,
            padded_colors.iter().cloned(),
        ),
        WriteDescriptorSet::buffer(BINDING_PALETTE_BUFFER, palette_buffer.clone()),
        WriteDescriptorSet::buffer(BINDING_GRID_INFO, grid_info_buffer.clone()),
    ];
    DescriptorSet::new(descriptor_set_allocator, layout, writes, [])
        .expect("Failed to create voxel descriptor set (array)")
}

/// Create a single 3D voxel image view and upload the provided packed voxel data.
pub fn create_voxel_image_view(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<
        vulkano::command_buffer::allocator::StandardCommandBufferAllocator,
    >,
    queue: Arc<Queue>, voxels: Vec<u128>, storage_w: u32, storage_h: u32, storage_d: u32,
) -> Arc<ImageView> {
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim3d,
            format: vulkano::format::Format::R32G32B32A32_UINT,
            extent: [storage_w, storage_h, storage_d],
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("Failed to create voxel image");

    let src_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        voxels,
    )
    .expect("Failed to create staging buffer");

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("Failed to create command buffer builder");
    command_buffer_builder
        .clear_color_image(ClearColorImageInfo::image(image.clone()))
        .unwrap()
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(src_buffer, image.clone()))
        .unwrap();
    let _ = command_buffer_builder.build().unwrap().execute(queue.clone()).unwrap();

    ImageView::new(image.clone(), vulkano::image::view::ImageViewCreateInfo::from_image(&image))
        .expect("Failed to create image view")
}

/// Create a 3D image view containing per-voxel palette indices (1 byte each).
/// Currently indices map to a procedural grayscale in the shader; later introduce a palette SSBO
/// (e.g. binding with an array of vec4 colors) for flexible material assignment without reallocating 3D images.
pub fn create_color_index_image_view(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<
        vulkano::command_buffer::allocator::StandardCommandBufferAllocator,
    >,
    queue: Arc<Queue>, indices: Vec<u8>, dim_x: u32, dim_y: u32, dim_z: u32,
) -> Arc<ImageView> {
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim3d,
            format: vulkano::format::Format::R8_UINT,
            extent: [dim_x, dim_y, dim_z],
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("Failed to create color index image");

    let src_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        indices,
    )
    .expect("Failed to create color index staging buffer");

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("Failed to create command buffer builder (color indices)");
    command_buffer_builder
        .clear_color_image(ClearColorImageInfo::image(image.clone()))
        .unwrap()
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(src_buffer, image.clone()))
        .unwrap();
    let _ = command_buffer_builder.build().unwrap().execute(queue.clone()).unwrap();

    ImageView::new(image.clone(), vulkano::image::view::ImageViewCreateInfo::from_image(&image))
        .expect("Failed to create color index image view")
}

/// Create a palette buffer (Vec<vec4>) with up to 256 colors. Returns subbuffer.
pub fn create_palette_buffer(
    memory_allocator: Arc<StandardMemoryAllocator>, colors: &[[f32; 4]],
) -> Subbuffer<[[f32; 4]]> {
    let usage = BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST;
    let buffer = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo { usage, ..Default::default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        colors.iter().cloned(),
    )
    .expect("Failed to create palette buffer");
    buffer
}

/// Create an empty placeholder voxel image view (all zeros) so the app can start immediately
/// before real voxel data is ready. This allocates the correctly sized 3D image for the current
/// resolution but does not perform any geometry processing.
pub fn create_empty_voxel_placeholder(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<
        vulkano::command_buffer::allocator::StandardCommandBufferAllocator,
    >,
    queue: Arc<Queue>, resolution: u32,
) -> Arc<ImageView> {
    // Placeholder uses legacy cubic layout until real data arrives.
    let voxel_texel_count = (resolution as usize).pow(3) / 128;
    let zeros = vec![0u128; voxel_texel_count.max(1)];
    create_voxel_image_view(
        memory_allocator,
        command_buffer_allocator,
        queue,
        zeros,
        resolution / 4,
        resolution / 4,
        resolution / 8,
    )
}

/// Create an empty placeholder color index 3D image (R8_UINT) matching full resolution.
pub fn create_empty_color_index_placeholder(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<
        vulkano::command_buffer::allocator::StandardCommandBufferAllocator,
    >,
    queue: Arc<Queue>, dim_x: u32, dim_y: u32, dim_z: u32,
) -> Arc<ImageView> {
    let total_voxels = (dim_x as usize).max(1) * (dim_y as usize).max(1) * (dim_z as usize).max(1);
    let zeros = vec![0u8; total_voxels.max(1)];
    create_color_index_image_view(
        memory_allocator,
        command_buffer_allocator,
        queue,
        zeros,
        dim_x,
        dim_y,
        dim_z,
    )
}

// (bulk creation helper removed as voxelization is now asynchronous per model)
