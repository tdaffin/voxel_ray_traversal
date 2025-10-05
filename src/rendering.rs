use std::sync::Arc;
use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::Device,
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageUsage,
        view::{ImageView, ImageViewCreateInfo},
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{ComputePipeline, Pipeline},
    swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo},
};
use winit::window::{Icon, Window};

pub struct RenderContext {
    pub window: Arc<Window>,
    pub swapchain: Arc<Swapchain>,
    pub image_views: Vec<Arc<ImageView>>,
    pub render_image: Arc<Image>,
    pub render_set: Arc<DescriptorSet>,
    pub resample_image: Arc<Image>,
    pub resample_set: Arc<DescriptorSet>,
    pub gui: egui_winit_vulkano::Gui,
    pub recreate_swapchain: bool,
}

pub fn get_allocators(
    device: &Arc<Device>,
) -> (
    Arc<StandardMemoryAllocator>,
    Arc<StandardDescriptorSetAllocator>,
    Arc<StandardCommandBufferAllocator>,
) {
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator =
        Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default()));
    let command_buffer_allocator =
        Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));
    (memory_allocator, descriptor_set_allocator, command_buffer_allocator)
}

pub fn get_swapchain_images(
    device: &Arc<Device>, surface: &Arc<Surface>, window: &Window,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let caps = device.physical_device().surface_capabilities(surface, Default::default()).unwrap();

    let image_format =
        device.physical_device().surface_formats(surface, Default::default()).unwrap()[0].0;

    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();

    // Determine a supported present mode, preferring Mailbox > Immediate > Fifo > FifoRelaxed.
    // FIFO is guaranteed to be supported by the Vulkan spec, so we fall back to it.
    let supported_present_modes =
        device.physical_device().surface_present_modes(surface, Default::default()).unwrap();
    let preferred =
        [PresentMode::Mailbox, PresentMode::Immediate, PresentMode::Fifo, PresentMode::FifoRelaxed];
    let present_mode = preferred
        .into_iter()
        .find(|m| supported_present_modes.contains(m))
        .unwrap_or(PresentMode::Fifo);

    // Choose image count (triple buffering if possible, otherwise clamp to supported range)
    let desired_image_count = 3u32.max(caps.min_image_count);
    let image_count = if let Some(max) = caps.max_image_count {
        desired_image_count.min(max)
    } else {
        desired_image_count
    };

    println!(
        "Creating swapchain with present mode: {:?} (supported: {:?})",
        present_mode, supported_present_modes
    );

    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: image_count,
            image_format,
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
            composite_alpha,
            present_mode,
            ..Default::default()
        },
    )
    .unwrap()
}

pub fn load_icon(icon: &[u8]) -> Icon {
    let (icon_rgba, icon_width, icon_height) = {
        let image = image::load_from_memory(icon).unwrap().to_rgba8();
        let (width, height) = image.dimensions();
        let rgba = image.into_raw();
        (rgba, width, height)
    };
    Icon::from_rgba(icon_rgba, icon_width, icon_height).unwrap()
}

pub fn get_render_image(
    memory_allocator: Arc<StandardMemoryAllocator>, extent: [u32; 2],
) -> (Arc<Image>, Arc<ImageView>) {
    let image = Image::new(
        memory_allocator,
        ImageCreateInfo {
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
            format: Format::R8G8B8A8_UNORM,
            extent: [extent[0], extent[1], 1],
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap();
    let image_view =
        ImageView::new(image.clone(), ImageViewCreateInfo::from_image(&image)).unwrap();
    (image, image_view)
}

pub fn get_resample_image(
    memory_allocator: Arc<StandardMemoryAllocator>, extent: [u32; 2],
) -> (Arc<Image>, Arc<ImageView>) {
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            format: Format::R8G8B8A8_UNORM,
            extent: [extent[0], extent[1], 1],
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap();
    let image_view =
        ImageView::new(image.clone(), ImageViewCreateInfo::from_image(&image)).unwrap();
    (image, image_view)
}

pub fn get_images_and_sets(
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    render_pipeline: &ComputePipeline, resample_pipeline: &ComputePipeline,
    render_extent: [u32; 2], window_extent: [u32; 2],
) -> (Arc<Image>, Arc<DescriptorSet>, Arc<Image>, Arc<DescriptorSet>) {
    let (render_image, render_image_view) =
        get_render_image(memory_allocator.clone(), render_extent);
    let layout = render_pipeline.layout().set_layouts()[0].clone();
    let render_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout,
        [WriteDescriptorSet::image_view(0, render_image_view.clone())],
        [],
    )
    .unwrap();
    let (resample_image, resample_image_view) = get_resample_image(memory_allocator, window_extent);
    let layout = resample_pipeline.layout().set_layouts()[0].clone();
    let resample_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout,
        [
            WriteDescriptorSet::image_view(0, render_image_view.clone()),
            WriteDescriptorSet::image_view(1, resample_image_view.clone()),
        ],
        [],
    )
    .unwrap();
    (render_image, render_set, resample_image, resample_set)
}
