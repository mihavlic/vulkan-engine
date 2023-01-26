#![allow(unused)]

use std::fs::File;
use std::{io, slice};

use graph::device::{self, DeviceCreateInfo, QueueFamilySelection};
use graph::graph::compile::GraphCompiler;
use graph::instance::{Instance, InstanceCreateInfo, OwnedInstance};
use graph::object::{self, ImageCreateInfo, PipelineStage, SwapchainCreateInfo};
use graph::passes::{self, ClearImage, SimpleShader};
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
        conf.add_extension(vk::KHR_DYNAMIC_RENDERING_EXTENSION_NAME);

        conf.fill_in_extensions();

        let info = InstanceCreateInfo {
            config: &mut conf,
            validation_layers: &[
                pumice::cstr!("VK_LAYER_KHRONOS_validation"),
                // pumice::cstr!("VK_LAYER_LUNARG_api_dump"),
            ],
            enable_debug_callback: true,
            app_name: "test application".to_owned(),
            verbose: false,
        };

        let instance = Instance::new(info);

        let surface = instance.create_surface(&window).unwrap();

        let mut sync = vk::PhysicalDeviceSynchronization2FeaturesKHR {
            synchronization_2: vk::TRUE,
            ..Default::default()
        };
        let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR {
            timeline_semaphore: vk::TRUE,
            p_next: (&mut sync) as *mut _ as *mut _,
            ..Default::default()
        };
        let dynamic = vk::PhysicalDeviceDynamicRenderingFeaturesKHR {
            dynamic_rendering: vk::TRUE,
            p_next: (&mut timeline) as *mut _ as *mut _,
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
            p_next: (&dynamic) as *const _ as *const _,
        };

        let device = device::Device::new(info);
        let queue = device.get_queue_bundle(0, 0).unwrap();

        let swapchain = make_swapchain(&window, surface, &device);

        let vert_spirv = read_spv(&mut File::open("target/shader.vert.spv").unwrap()).unwrap();
        let frag_spirv = read_spv(&mut File::open("target/shader.frag.spv").unwrap()).unwrap();

        let vert_module = device.create_shader_module(vert_spirv).unwrap();
        let frag_module = device.create_shader_module(frag_spirv).unwrap();

        let empty_layout = device
            .create_pipeline_layout(object::PipelineLayoutCreateInfo::empty())
            .unwrap();

        let pipeline_info = object::GraphicsPipelineCreateInfo::builder()
            .stages([
                PipelineStage {
                    flags: vk::PipelineShaderStageCreateFlags::empty(),
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: vert_module,
                    name: "main".into(),
                    specialization_info: None,
                },
                PipelineStage {
                    flags: vk::PipelineShaderStageCreateFlags::empty(),
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    module: frag_module,
                    name: "main".into(),
                    specialization_info: None,
                },
            ])
            .input_assembly(object::state::InputAssembly {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                primitive_restart_enable: false,
            })
            .viewport(object::state::Viewport::default())
            .vertex_input(object::state::VertexInput {
                vertex_bindings: Vec::new(),
                vertex_attributes: Vec::new(),
            })
            .rasterization(object::state::Rasterization {
                depth_clamp_enable: false,
                rasterizer_discard_enable: false,
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            })
            .multisample(object::state::Multisample {
                rasterization_samples: vk::SampleCountFlags::C1,
                ..Default::default()
            })
            .color_blend(object::state::ColorBlend {
                attachments: vec![object::state::Attachment {
                    color_write_mask: vk::ColorComponentFlags::all(),
                    ..Default::default()
                }],
                ..Default::default()
            })
            .dynamic_state([vk::DynamicState::SCISSOR, vk::DynamicState::VIEWPORT])
            .layout(empty_layout)
            .finish();

        let pipeline = device.create_delayed_pipeline(pipeline_info);

        let mut compiler = GraphCompiler::new();
        let mut graph = compiler.compile(device.clone(), device.threadpool(), |b| {
            let queue = b.import_queue(queue);
            let swapchain = b.acquire_swapchain(swapchain.clone());
            b.add_pass(
                queue,
                ClearImage {
                    image: swapchain,
                    color: vk::ClearColorValue {
                        float_32: [1.0, 0.0, 1.0, 1.0],
                    },
                },
                "Swapchain clear",
            );
            b.add_pass(
                queue,
                SimpleShader {
                    pipeline,
                    attachments: vec![swapchain],
                },
                "",
            );
        });

        event_loop.run(move |event, _, control_flow| {
            control_flow.set_poll();

            match event {
                Event::WindowEvent { event, window_id } => match event {
                    WindowEvent::CloseRequested if window_id == window.id() => {
                        control_flow.set_exit()
                    }
                    WindowEvent::Resized(size) => {
                        let extent = vk::Extent2D {
                            width: size.width,
                            height: size.height,
                        };
                        swapchain.surface_resized(extent);
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(req) => {
                    graph.run();
                    device.idle_cleanup_poll();
                }
                _ => (),
            }
        });
    }
}

unsafe fn make_swapchain(
    window: &winit::window::Window,
    surface: object::Surface,
    device: &device::OwnedDevice,
) -> object::Swapchain {
    let extent = {
        let size = window.inner_size();
        vk::Extent2D {
            width: size.width,
            height: size.height,
        }
    };

    // TODO swapchain configuration fallback for formats, present modes, and color spaces
    let info = SwapchainCreateInfo {
        surface: surface.into_raw(),
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

    device.create_swapchain(info).unwrap()
}

// stolen from ash
pub fn read_spv<R: io::Read + io::Seek>(x: &mut R) -> io::Result<Vec<u32>> {
    let size = x.seek(io::SeekFrom::End(0))?;
    if size % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input length not divisible by 4",
        ));
    }
    if size > usize::max_value() as u64 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "input too long"));
    }
    let words = (size / 4) as usize;
    // https://github.com/MaikKlein/ash/issues/354:
    // Zero-initialize the result to prevent read_exact from possibly
    // reading uninitialized memory.
    let mut result = vec![0u32; words];
    x.seek(io::SeekFrom::Start(0))?;
    x.read_exact(unsafe {
        slice::from_raw_parts_mut(result.as_mut_ptr().cast::<u8>(), words * 4)
    })?;
    const MAGIC_NUMBER: u32 = 0x0723_0203;
    if !result.is_empty() && result[0] == MAGIC_NUMBER.swap_bytes() {
        for word in &mut result {
            *word = word.swap_bytes();
        }
    }
    if result.is_empty() || result[0] != MAGIC_NUMBER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input missing SPIR-V magic number",
        ));
    }
    Ok(result)
}
