#![allow(unused)]

use graph::context::device::{DeviceCreateInfo, QueueFamilySelection};
use graph::context::instance::{Instance, InstanceCreateInfo};
use graph::object::{self, SwapchainCreateInfo};
use pumice::{util::config::ApiLoadConfig, vk};
use pumice_vma::{AllocationCreateFlags, AllocationCreateInfo};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::WindowBuilder;

fn main() {
    unsafe {
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

        let info = InstanceCreateInfo {
            config: &mut conf,
            validation_layers: &[pumice::cstr!("VK_LAYER_KHRONOS_validation")],
            enable_debug_callback: true,
            app_name: pumice::cstr!("test_context_new"),
            verbose: false,
        };

        let instance = Instance::new(info);

        let surface = instance.create_surface(&window).unwrap();

        let info = DeviceCreateInfo {
            instance,
            config: &mut conf,
            device_features: Default::default(),
            queue_family_selection: &[QueueFamilySelection {
                mask: vk::QueueFlags::GRAPHICS,
                count: 1,
                priority: 1.0,
                exact: false,
                attempt_dedicated: false,
                coalesce: true,
                support_surfaces: &[&surface],
            }],
            device_substrings: &[],
            verbose: false,
        };

        let device = graph::context::device::Device::new(info);

        let extent = {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width,
                height: size.height,
            }
        };

        let info = SwapchainCreateInfo {
            surface: surface.to_raw(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            min_image_count: 2,
            format: vk::Format::R8G8B8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            extent,
            array_layers: 1,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: vk::PresentModeKHR::FIFO,
            clipped: false,
        };

        // let swapchain = device.create_swapchain(info).unwrap();

        // device
        //     .device()
        //     .get_swapchain_images_khr(swapchain, swapchain_image_count);

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
                _ => (),
            }
        });

        drop(device);
    }
}
