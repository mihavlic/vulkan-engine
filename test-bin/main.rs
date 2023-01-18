#![allow(unused)]

use graph::device::{self, DeviceCreateInfo, QueueFamilySelection};
use graph::graph::passes::ClearImage;
use graph::graph::Graph;
use graph::instance::{Instance, InstanceCreateInfo};
use graph::object::{self, ImageCreateInfo, SwapchainCreateInfo};
use graph::tracing::tracing_subscriber::install_tracing_subscriber;
use pumice::{util::ApiLoadConfig, vk};
use pumice_vma::{AllocationCreateFlags, AllocationCreateInfo};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::WindowBuilder;

fn main() {
    unsafe {
        install_tracing_subscriber(None);
        let mut event_loop = EventLoop::new();

        let window = WindowBuilder::new()
            .with_title("A fantastic window!")
            .with_inner_size(winit::dpi::LogicalSize::new(128.0f32, 128.0f32))
            .build(&event_loop)
            .unwrap();

        let mut conf = ApiLoadConfig::new(vk::API_VERSION_1_0);
        conf.add_extensions_iter(
            pumice::surface::enumerate_required_extensions(&window)
                .unwrap()
                .into_iter()
                .cloned(),
        );
        conf.add_extension(vk::KHR_SWAPCHAIN_EXTENSION_NAME);
        conf.add_extension(vk::KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
        conf.add_extension(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        conf.add_extension(vk::KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

        let info = InstanceCreateInfo {
            config: &mut conf,
            validation_layers: &[pumice::cstr!("VK_LAYER_KHRONOS_validation")],
            enable_debug_callback: true,
            app_name: pumice::cstr!("test application"),
            verbose: false,
        };

        let instance = Instance::new(info);

        let surface = instance.create_surface(&window).unwrap();

        let mut sync = vk::PhysicalDeviceSynchronization2FeaturesKHR {
            synchronization_2: vk::TRUE,
            ..Default::default()
        };
        let timeline = vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR {
            timeline_semaphore: vk::TRUE,
            p_next: (&mut sync) as *mut _ as *mut _,
            ..Default::default()
        };
        let info = DeviceCreateInfo {
            instance,
            config: &mut conf,
            device_features: vk::PhysicalDeviceFeatures {
                ..Default::default()
            },
            queue_family_selection: &[QueueFamilySelection {
                mask: vk::QueueFlags::GRAPHICS,
                count: 1,
                priority: 1.0,
                exact: false,
                attempt_dedicated: false,
                coalesce: true,
                support_surfaces: &[&surface],
            }],
            device_substrings: &["NVIDIA"],
            verbose: false,
            p_next: (&timeline) as *const _ as *const _,
        };

        let device = device::Device::new(info);
        let queue = device.get_queue_bundle(0, 0).unwrap();

        let extent = {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width,
                height: size.height,
            }
        };

        // let format = {
        //     let formats = instance.handle().get_physical_device_surface_formats_khr(
        //         device.physical_device(),
        //         surface.handle(),
        //         None,
        //     );
        // };

        let info = SwapchainCreateInfo {
            surface: surface.to_raw(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            min_image_count: 2,
            format: vk::Format::B8G8R8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            extent,
            array_layers: 1,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: vk::PresentModeKHR::FIFO,
            clipped: false,
        };

        let swapchain = device.create_swapchain(info).unwrap();
        let mut graph = Graph::new(device.clone());

        event_loop.run_return(move |event, _, control_flow| {
            control_flow.set_poll();

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == window.id() => control_flow.set_exit(),
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(req) => {
                    graph.run(|b| {
                        let queue = b.import_queue(queue);
                        let swapchain = b.acquire_swapchain(swapchain.clone());
                        b.add_pass(
                            queue,
                            ClearImage {
                                image: swapchain,
                                color: vk::ClearColorValue {
                                    uint_32: [255, 0, 255, 0],
                                },
                            },
                            "Swapchain clear",
                        );
                    });
                    device.idle_cleanup_poll();
                }
                _ => (),
            }
        });
    }
}
