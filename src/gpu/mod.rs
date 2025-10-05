use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::QueueFlags;
use vulkano::device::physical::PhysicalDevice;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::{
    Version, VulkanLibrary,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    swapchain::Surface,
};
use winit::event_loop::EventLoop;

use crate::rendering::get_allocators;

#[derive(Debug)]
pub struct GpuContext {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl GpuContext {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let instance = create_instance(event_loop);
        let (physical_device, queue_family_index, mut device_extensions) =
            select_physical_and_queue(&instance, event_loop);
        let (device, queue) =
            create_device(physical_device, queue_family_index, &mut device_extensions);
        let (memory_allocator, descriptor_set_allocator, command_buffer_allocator) =
            get_allocators(&device);

        Self {
            instance,
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
        }
    }
}

fn create_instance(event_loop: &EventLoop<()>) -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("Failed to load Vulkan library");
    let mut required_extensions =
        Surface::required_extensions(event_loop).expect("Surface extensions");
    required_extensions.ext_debug_utils = true; // keep for validation/debug overlays
    Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("Instance creation failed")
}

fn select_physical_and_queue(
    instance: &Arc<Instance>, event_loop: &EventLoop<()>,
) -> (Arc<PhysicalDevice>, u32, DeviceExtensions) {
    let mut device_extensions =
        DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::empty() };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .expect("Failed to enumerate physical devices")
        .filter(|p| {
            p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
        })
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.presentation_support(i as u32, event_loop).unwrap()
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            // scoring preference
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("No suitable physical device found");

    if physical_device.api_version() < Version::V1_3 {
        device_extensions.khr_dynamic_rendering = true;
    }

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    (physical_device, queue_family_index, device_extensions)
}

fn create_device(
    physical_device: Arc<PhysicalDevice>, queue_family_index: u32,
    device_extensions: &mut DeviceExtensions,
) -> (Arc<Device>, Arc<Queue>) {
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo { queue_family_index, ..Default::default() }],
            enabled_extensions: *device_extensions,
            enabled_features: DeviceFeatures { dynamic_rendering: true, ..DeviceFeatures::empty() },
            ..Default::default()
        },
    )
    .expect("Device creation failed");

    let queue = queues.next().expect("Expected at least one queue");
    (device, queue)
}
