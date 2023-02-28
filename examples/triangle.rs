#![allow(unused)]

use std::env::current_dir;
use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::{io, slice};

use graph::device::reflection::{ReflectedLayout, SpirvModule};
use graph::device::{self, read_spirv, DeviceCreateInfo, QueueFamilySelection};
use graph::graph::compile::GraphCompiler;
use graph::graph::execute::GraphRunConfig;
use graph::instance::{Instance, InstanceCreateInfo, OwnedInstance};
use graph::object::{self, ImageCreateInfo, PipelineStage, SwapchainCreateInfo};
use graph::passes::{self, ClearImage, SimpleShader};
use graph::tracing::tracing_subscriber::install_tracing_subscriber;
use pumice::{util::ApiLoadConfig, vk};
use pumice_vma::{AllocationCreateFlags, AllocationCreateInfo};
use smallvec::{smallvec, SmallVec};
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
        conf.add_extension(vk::EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);

        conf.fill_in_extensions();

        let info = InstanceCreateInfo {
            config: &mut conf,
            validation_layers: &[
                // pumice::cstr!("VK_LAYER_KHRONOS_validation"),
                // pumice::cstr!("VK_LAYER_LUNARG_api_dump"),
            ],
            enable_debug_callback: true,
            debug_labeling: true,
            app_name: "test application".to_owned(),
            verbose: false,
        };

        let instance = Instance::new(info);

        let surface = instance.create_surface(&window).unwrap();

        let mut scalar_layout = vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT {
            scalar_block_layout: vk::TRUE,
            ..Default::default()
        };
        let mut sync = vk::PhysicalDeviceSynchronization2FeaturesKHR {
            synchronization_2: vk::TRUE,
            p_next: (&mut scalar_layout) as *mut _ as *mut _,
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

        fn checked_file_open(path: impl AsRef<Path>) -> File {
            File::open(path.as_ref()).unwrap_or_else(|_| {
                panic!(
                    "Error opening file at {}\ncwd: {}",
                    path.as_ref().to_string_lossy(),
                    current_dir().unwrap().to_string_lossy()
                )
            })
        }

        // let vert_module = device
        //     .create_shader_module_read(&mut checked_file_open("examples/shader.vert.spv"))
        //     .unwrap();
        // let frag_module = device
        //     .create_shader_module_read(&mut checked_file_open("examples/shader.frag.spv"))
        //     .unwrap();

        let mut vert_bytes = Cursor::new(include_bytes!("shader.vert.spv").as_slice());
        let mut frag_bytes = Cursor::new(include_bytes!("shader.frag.spv").as_slice());

        let vert_spirv = read_spirv(&mut vert_bytes).unwrap();
        let frag_spirv = read_spirv(&mut frag_bytes).unwrap();

        let vert_module = device.create_shader_module_spirv(&vert_spirv).unwrap();
        let frag_module = device.create_shader_module_spirv(&frag_spirv).unwrap();

        let pipeline_layout = ReflectedLayout::new(&[
            SpirvModule {
                spirv: &vert_spirv,
                entry_points: &["main"],
                dynamic_uniform_buffers: true,
                dynamic_storage_buffers: false,
                include_unused_descriptors: false,
            },
            SpirvModule {
                spirv: &frag_spirv,
                entry_points: &["main"],
                dynamic_uniform_buffers: true,
                dynamic_storage_buffers: false,
                include_unused_descriptors: false,
            },
        ])
        .create(&device, vk::DescriptorSetLayoutCreateFlags::empty())
        .unwrap();

        let set_layout = pipeline_layout.get_descriptor_set_layouts()[0].clone();

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
            .viewport(object::state::Viewport {
                // the actual contents are ignored, it is just important to have one for each
                viewports: smallvec![Default::default()],
                scissors: smallvec![Default::default()],
            })
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
            .layout(pipeline_layout.clone())
            .finish();

        let pipeline = device.create_delayed_graphics_pipeline(pipeline_info);

        let mut compiler = GraphCompiler::new();
        let graph = compiler.compile(device.clone(), |b| {
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
                    set_layout,
                    pipeline_layout,
                    pipeline,
                    attachments: vec![swapchain],
                },
                "",
            );
        });

        let mut graph = Some(graph);
        let mut device = Some(device);
        let mut swapchain = Some(swapchain);

        event_loop.run_return(move |event, _, control_flow| {
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
                        swapchain.as_ref().unwrap().surface_resized(extent);
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(req) => {
                    let status = graph.as_mut().unwrap().run(GraphRunConfig {
                        swapchain_acquire_timeout_ns: 1000_000_000 / 2,
                        acquire_swapchain_with_fence: false,
                        return_after_swapchain_recreate: false,
                    });
                    device.as_ref().unwrap().idle_cleanup_poll();
                }
                Event::LoopDestroyed => {
                    drop(graph.take());
                    drop(swapchain.take());
                    let device = device.take().unwrap();
                    device.drain_work();
                    device.attempt_destroy().unwrap();
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

    let (present_modes, result) = device
        .instance()
        .handle()
        .get_physical_device_surface_present_modes_khr(
            device.physical_device(),
            surface.handle(),
            None,
        )
        .unwrap();
    assert_eq!(result, vk::Result::SUCCESS);

    let mut present_mode = vk::PresentModeKHR::FIFO;
    for mode in present_modes {
        if mode == vk::PresentModeKHR::MAILBOX {
            present_mode = vk::PresentModeKHR::MAILBOX;
            break;
        }
    }

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
        present_mode,
        clipped: false,
    };

    device.create_swapchain(info).unwrap()
}
