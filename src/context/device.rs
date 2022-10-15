use std::{
    collections::{hash_map::RandomState, HashSet},
    ffi::CStr,
    fmt::Display,
    pin::Pin,
    sync::Mutex,
};

use crate::{
    object::{self},
    storage::{nostore::NoStore, GetContextStorage, ObjectStorage},
    tracing::shim_macros::{info, trace},
    util::{
        self,
        debug_callback::to_version,
        format_utils::{self, Fun, IterDisplay},
    },
};
use pumice::{
    loader::{
        tables::{DeviceTable, EntryTable, InstanceTable},
        InstanceLoader,
    },
    util::result::VulkanResult,
    vk,
    vk10::QueueFamilyProperties,
    DeviceWrapper,
};

#[derive(Debug, Clone, Copy)]
pub enum SelectionMechanism {
    Exact,
    Contains,
}

#[derive(Debug, Clone)]
pub struct QueueFamilySelection {
    mask: vk::QueueFlags,
    count: u32,
    priority: f32,
    exact: bool,
    attempt_dedicated: bool,
    coalesce: bool,
}

pub struct ContextCreateInfo<'a> {
    conf: pumice::util::config::ApiLoadConfig<'a>,
    device_features: vk::PhysicalDeviceFeatures,
    validation_layers: &'a [&'a CStr],
    enable_debug_callback: bool,
    verbose: bool,
    queue_family_selection: Vec<QueueFamilySelection>,
    app_name: &'a CStr,
}

pub(crate) struct InnerDevice {
    pub(crate) device: pumice::DeviceWrapper,

    pub(crate) physical_device_properties: vk::PhysicalDeviceProperties,
    pub(crate) physical_device_features: vk::PhysicalDeviceFeatures,

    // family index, range for the subslice of the requested queues
    pub(crate) queue_selection_mapping: Vec<(usize, std::ops::Range<usize>)>,
    pub(crate) queues: Vec<vk::Queue>,
    pub(crate) queue_families: Vec<QueueFamilyProperties>,

    pub(crate) debug_messenger: Option<pumice::extensions::ext_debug_utils::DebugUtilsMessengerEXT>,
    pub(crate) verbose: bool,

    // hanobjectdle storage
    pub(crate) image_storage: NoStore,

    // allocator
    pub(crate) allocator: pumice_vma::Allocator,
    pub(crate) allocation_callbacks: Option<vk::AllocationCallbacks>,

    // at the bottom so that these are dropped last, the loader keeps the vulkan dll loaded
    pub(crate) device_table: DeviceTable,
}

pub struct Device(Pin<Box<InnerDevice>>);

impl Device {
    pub unsafe fn new(info: ContextCreateInfo) -> Self {
        let allocation_callbacks = None;

        let ContextCreateInfo {
            mut conf,
            device_features,
            validation_layers,
            enable_debug_callback,
            verbose,
            queue_family_selection,
            app_name,
        } = info;

        if enable_debug_callback {
            conf.add_extension(pumice::extensions::ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        let mut tables = Box::new((
            EntryTable::new_empty(),
            InstanceTable::new_empty(),
            DeviceTable::new_empty(),
        ));

        let loader = pumice::loader::EntryLoader::new().expect("Failed to create entry loader");
        tables.0.load(&loader);
        let entry = pumice::EntryWrapper::new(&tables.0);

        let api_version = entry
            .enumerate_instance_version()
            .expect("enumerate_instance_version error");

        if api_version < conf.get_api_version() {
            let requested = to_version(conf.get_api_version());
            let api_version = to_version(api_version);
            panic!(
                "Unsupported api version, requested {}.{}.{}, available {}.{}.{}",
                requested.0, requested.1, requested.2, api_version.0, api_version.1, api_version.2
            );
        }

        {
            let available_instance_extensions = entry
                .enumerate_instance_extension_properties(None, None)
                .expect("enumerate_instance_extension_properties error");

            let available_instance_extensions: HashSet<&CStr, RandomState> = HashSet::from_iter(
                available_instance_extensions
                    .iter()
                    .map(|p| CStr::from_ptr(p.extension_name.as_ptr())),
            );

            let missing_extensions = conf
                .get_instance_extensions_iter()
                .filter(|e| !available_instance_extensions.contains(e));

            if missing_extensions.clone().next().is_some() {
                let missing =
                    IterDisplay::new(missing_extensions, |i, d| i.to_string_lossy().fmt(d));
                panic!("Instance doesn't support requested extensions:\n{missing}");
            }
        }

        {
            let available_layers = entry
                .enumerate_instance_layer_properties(None)
                .expect("enumerate_instance_extension_properties error");

            let available_layers: HashSet<&CStr, RandomState> = HashSet::from_iter(
                available_layers
                    .iter()
                    .map(|p| CStr::from_ptr(p.layer_name.as_ptr())),
            );

            let missing_extensions = validation_layers
                .iter()
                .filter(|&&e| !available_layers.contains(e));

            if missing_extensions.clone().next().is_some() {
                let missing =
                    IterDisplay::new(missing_extensions, |i, d| i.to_string_lossy().fmt(d));
                panic!("Instance doesn't support requested layers:\n{missing}");
            }
        }

        let instance_layer_properties = entry
            .enumerate_instance_layer_properties(None)
            .expect("enumerate_instance_layer_properties error");

        if verbose {
            let iter = instance_layer_properties
                .iter()
                .map(|l| CStr::from_ptr(l.layer_name.as_ptr()).to_string_lossy());
            let iter = IterDisplay::new(iter, |i, w| i.fmt(w));
            info!(
                "{} instance layers:\n{}",
                instance_layer_properties.len(),
                iter
            );
        } else {
            info!("{} instance layers", instance_layer_properties.len());
        }

        let app_info = pumice::vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: 0,
            p_engine_name: app_name.as_ptr(),
            engine_version: 0,
            api_version: conf.get_api_version(),
            ..Default::default()
        };

        let instance_extensions = conf.get_instance_extensions();

        let validation_layers = validation_layers
            .iter()
            .map(|l| l.as_ptr())
            .collect::<Vec<_>>();

        let create_info = pumice::vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_layer_count: validation_layers.len() as _,
            pp_enabled_layer_names: validation_layers.as_ptr(),
            enabled_extension_count: instance_extensions.len() as _,
            pp_enabled_extension_names: instance_extensions.as_ptr(),
            ..Default::default()
        };

        let instance_handle = entry
            .create_instance(&create_info, allocation_callbacks.as_ref())
            .expect("create_instance error");
        let instance_loader = InstanceLoader::new(instance_handle, &loader);
        tables.1.load(&instance_loader, &conf);
        tables.2.load(&instance_loader, &conf);

        let instance = pumice::InstanceWrapper::new(instance_handle, &tables.1);

        let debug_messenger = if enable_debug_callback {
            let info = pumice::extensions::ext_debug_utils::DebugUtilsMessengerCreateInfoEXT {
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::all(),
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                pfn_user_callback: Some(util::debug_callback::debug_callback),
                ..Default::default()
            };

            Some(
                instance
                    .create_debug_utils_messenger_ext(&info, allocation_callbacks.as_ref())
                    .expect("create_debug_utils_messenger_ext error"),
            )
        } else {
            None
        };

        let physical_devices = instance
            .enumerate_physical_devices(None)
            .expect("enumerate_physical_devices error");
        let physical_device_properties = physical_devices
            .iter()
            .map(|&d| instance.get_physical_device_properties(d))
            .collect::<Vec<_>>();

        if verbose {
            let fun = Fun::new(|f: &mut std::fmt::Formatter<'_>| {
                for (i, property) in physical_device_properties.iter().enumerate() {
                    let name = CStr::from_ptr(property.device_name.as_ptr()).to_string_lossy();
                    let core = to_version(property.api_version);
                    let driver = to_version(property.driver_version);
                    write! {
                        f,
                        "\n[{i}] {name}\n    core {}.{}.{}, driver {}.{}.{}, {:?}",
                        core.0, core.1, core.2,
                        driver.0, driver.1, driver.2,
                        property.device_type,
                    }?;
                }
                Ok(())
            });
            info!("{} physical devices:{}", physical_devices.len(), fun);
        } else {
            info!("{} physical devices", physical_devices.len());
        }

        // (_, _, overlay over queue_family_selection and their associated queue families)
        let (
            physical_device,
            physical_device_properties,
            physical_device_features,
            queue_families,
            selected_queue_families,
        ) = select_device(
            &physical_devices,
            &physical_device_properties,
            &instance,
            &conf,
            &queue_family_selection,
        );

        // (queue family, individual queue priorities, offset of reclaimed queue (needed later))
        let mut queue_creations = Vec::new();

        for (selection, selected_family) in
            queue_family_selection.iter().zip(&selected_queue_families)
        {
            let found = queue_creations
                .iter_mut()
                .find(|(family, _, _)| *family == *selected_family);

            let priorities = if let Some(found) = found {
                &mut found.1
            } else {
                queue_creations.push((*selected_family, Vec::new(), 0));
                &mut queue_creations.last_mut().unwrap().1
            };

            priorities.push(selection.priority);
        }

        let queue_create_infos = queue_creations
            .iter()
            .map(
                |(family, priorities, _)| pumice::vk::DeviceQueueCreateInfo {
                    queue_family_index: *family as _,
                    queue_count: priorities.len() as _,
                    p_queue_priorities: priorities.as_ptr(),
                    ..Default::default()
                },
            )
            .collect::<Vec<_>>();

        let device_extensions = conf.get_device_extensions();

        let device_create_info = pumice::vk::DeviceCreateInfo {
            queue_create_info_count: queue_create_infos.len() as _,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_extension_count: device_extensions.len() as _,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            p_enabled_features: &device_features,
            ..Default::default()
        };

        let device_handle = instance
            .create_device(
                physical_device,
                &device_create_info,
                allocation_callbacks.as_ref(),
            )
            .expect("create_device error");

        let device = DeviceWrapper::new(device_handle, &tables.2);

        let (queues, queue_selection_mapping) = {
            let mut queues = Vec::new();
            let mut queue_selection_mapping = Vec::new();

            for (selection, selected_family) in
                queue_family_selection.iter().zip(&selected_queue_families)
            {
                queue_selection_mapping.push((
                    *selected_family,
                    queues.len()..queues.len() + selection.count as usize,
                ));

                for _ in 0..selection.count {
                    let (_, _, offset) = queue_creations
                        .iter_mut()
                        .find(|(family, _, _)| *family == *selected_family)
                        .unwrap();

                    queues.push(device.get_device_queue(*selected_family as _, *offset));

                    *offset += 1;
                }
            }

            (queues, queue_selection_mapping)
        };

        let allocator = {
            let create_info =
                pumice_vma::AllocatorCreateInfo::new(&instance, &device, &physical_device);
            pumice_vma::Allocator::new(create_info).unwrap()
        };

        let inner = InnerDevice {
            entry,
            instance,

            device,
            physical_device_properties,
            physical_device_features,

            queue_families,
            queue_selection_mapping,
            queues,

            debug_messenger,
            verbose,

            image_storage: NoStore::new(),

            allocator,
            allocation_callbacks,

            loader,
            device_table: tables,
        };

        Self(Box::pin(inner))
    }
    pub fn entry(&self) -> &pumice::EntryWrapper {
        &self.0.entry
    }
    pub fn instance(&self) -> &pumice::InstanceWrapper {
        &self.0.instance
    }
    pub fn device(&self) -> &pumice::DeviceWrapper {
        &self.0.device
    }
    pub fn allocator(&self) -> &pumice_vma::Allocator {
        &self.0.allocator
    }
    pub fn allocator_callbacks(&self) -> Option<&vk::AllocationCallbacks> {
        self.0.allocation_callbacks.as_ref()
    }
    pub fn get_queue(&self, selection_index: usize, offset: usize) -> Option<vk::Queue> {
        let range = self
            .0
            .queue_selection_mapping
            .get(selection_index)?
            .1
            .clone();
        self.0.queues.get(range)?.get(offset).cloned()
    }
    pub fn get_queue_family(&self, selection_index: usize, offset: usize) -> Option<u32> {
        self.0
            .queue_selection_mapping
            .get(selection_index)
            .map(|(family, _)| *family as u32)
    }

    pub unsafe fn create_image(
        &self,
        info: object::ImageCreateInfo,
        allocate: pumice_vma::AllocationCreateInfo,
    ) -> VulkanResult<object::Image> {
        self.0
            .image_storage
            .get_or_create(info, allocate, &self.0)
            .map(object::Image)
    }

    fn inner(&self) -> &InnerDevice {
        &self.0
    }
}

#[cfg(test)]
use {
    crate::tracing::tracing_subscriber::install_tracing_subscriber,
    crate::tracing::Severity,
    pumice_vma::{AllocationCreateFlags, AllocationCreateInfo},
    std::str::FromStr,
};

#[test]
fn test_context_new() {
    install_tracing_subscriber(Severity::Trace);

    let info = ContextCreateInfo {
        conf: pumice::util::config::ApiLoadConfig::new(vk::API_VERSION_1_0),
        device_features: Default::default(),
        validation_layers: &[pumice::cstr!("VK_LAYER_KHRONOS_validation")],
        enable_debug_callback: true,
        verbose: false,
        queue_family_selection: vec![QueueFamilySelection {
            mask: vk::QueueFlags::GRAPHICS,
            count: 1,
            priority: 1.0,
            exact: false,
            attempt_dedicated: false,
            coalesce: true,
        }],
        app_name: pumice::cstr!("test_context_new"),
    };

    unsafe {
        let context = Device::new(info);
    }
}

#[test]
fn test_create_image() {
    install_tracing_subscriber(Severity::Info);

    let info = ContextCreateInfo {
        conf: pumice::util::config::ApiLoadConfig::new(vk::API_VERSION_1_0),
        device_features: Default::default(),
        validation_layers: &[pumice::cstr!("VK_LAYER_KHRONOS_validation")],
        enable_debug_callback: true,
        verbose: false,
        queue_family_selection: vec![QueueFamilySelection {
            mask: vk::QueueFlags::GRAPHICS,
            count: 1,
            priority: 1.0,
            exact: false,
            attempt_dedicated: false,
            coalesce: true,
        }],
        app_name: pumice::cstr!("test_context_new"),
    };

    unsafe {
        let context = Device::new(info);

        let info = object::ImageCreateInfo {
            flags: vk::ImageCreateFlags::empty(),
            size: object::Extent::D2(1, 1),
            format: vk::Format::R8G8B8A8_SRGB,
            samples: vk::SampleCountFlags::C1,
            mip_levels: 1,
            array_layers: 1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            initial_layout: vk::ImageLayout::UNDEFINED,
        };
        let allocation_info = AllocationCreateInfo::new()
            .required_flags(vk::MemoryPropertyFlags::HOST_VISIBLE)
            .preferred_flags(
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED,
            )
            .flags(AllocationCreateFlags::MAPPED);
        let image = context.create_image(info, allocation_info);

        // explicit drop order due to very messy code
        drop(image);
        drop(context);
    }
}

unsafe fn select_device(
    physical_devices: &Vec<vk::PhysicalDevice>,
    physical_device_properties: &Vec<vk::PhysicalDeviceProperties>,
    instance: &pumice::InstanceWrapper,
    conf: &pumice::util::config::ApiLoadConfig,
    queue_family_selection: &Vec<QueueFamilySelection>,
) -> (
    vk::PhysicalDevice,
    vk::PhysicalDeviceProperties,
    vk::PhysicalDeviceFeatures,
    Vec<QueueFamilyProperties>,
    Vec<usize>,
) {
    let mut chosen_index = None;
    let mut selected_queue_families = Vec::new();
    let mut queue_families = Vec::new();

    let mut scratch_extensions = HashSet::new();
    let mut family_search_scratch = Vec::new();

    let device_extensions: HashSet<&CStr, RandomState> =
        HashSet::from_iter(conf.get_device_extensions_iter());

    for (i, (physical_device, physical_device_properties)) in physical_devices
        .iter()
        .cloned()
        .zip(physical_device_properties)
        .enumerate()
    {
        let device_name =
            CStr::from_ptr(physical_device_properties.device_name.as_ptr()).to_string_lossy();

        // extension criteria
        {
            let extensions = instance
                .enumerate_device_extension_properties(physical_device, None, None)
                .expect("enumerate_device_extension_properties error");

            scratch_extensions.clear();
            scratch_extensions.extend(
                extensions
                    .iter()
                    .map(|e| CStr::from_ptr(e.extension_name.as_ptr())),
            );

            let difference = device_extensions.difference(&scratch_extensions);

            if difference.clone().next().is_some() {
                let iter =
                    format_utils::IterDisplay::new(difference, |i, d| i.to_string_lossy().fmt(d));

                trace!("Device '{device_name}' is missing extensions:\n{iter}");
                info!("Device '{}' skipped due to missing extensions", device_name);

                continue;
            }
        }

        // queue family criteria
        {
            queue_families =
                instance.get_physical_device_queue_family_properties(physical_device, None);

            selected_queue_families.clear();
            family_search_scratch.clear();
            family_search_scratch.resize(queue_families.len(), 0);

            for (i, selection) in queue_family_selection.iter().enumerate() {
                let is_valid: fn(vk::QueueFlags, vk::QueueFlags) -> bool = if selection.exact {
                    |flags: vk::QueueFlags, mask: vk::QueueFlags| flags == mask
                } else {
                    |flags: vk::QueueFlags, mask: vk::QueueFlags| flags.contains(mask)
                };

                let family_iter = queue_families.iter().zip(&family_search_scratch);

                // try to select some family that does not have queues
                if selection.attempt_dedicated {
                    let position = family_iter.clone().position(|(family, selected_count)| {
                        *selected_count == 0
                            && is_valid(family.queue_flags, selection.mask)
                            && *selected_count + selection.count <= family.queue_count
                    });

                    if let Some(position) = position {
                        family_search_scratch[position] += selection.count;
                        selected_queue_families.push(position);
                        continue;
                    }
                }
                // try to select some family that already has queues
                if selection.coalesce {
                    let position = family_iter.clone().position(|(family, selected_count)| {
                        *selected_count > 0
                            && is_valid(family.queue_flags, selection.mask)
                            && *selected_count + selection.count <= family.queue_count
                    });

                    if let Some(position) = position {
                        family_search_scratch[position] += selection.count;
                        selected_queue_families.push(position);
                        continue;
                    }
                }
                // then try to match anything
                {
                    let position = family_iter.clone().position(|(family, selected_count)| {
                        is_valid(family.queue_flags, selection.mask)
                            && *selected_count + selection.count <= family.queue_count
                    });

                    if let Some(position) = position {
                        family_search_scratch[position] += selection.count;
                        selected_queue_families.push(position);
                        continue;
                    }
                }

                info!(
                    "Device '{}' skipped because it couldn't satisfy queue selection:\n{:?}",
                    device_name, selection
                );
            }
        }

        // {
        //     physical_device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
        //     ...
        // }

        // TODO image formats, swapchain present modes, device limits, device features, possibly memory heaps
        // {
        //     instance.get_physical_device_features(physical_device)
        //     ...
        // }

        info!("Selected device '{}'", device_name);
        chosen_index = Some(i);
        break;
    }

    let chosen_index = chosen_index.expect("Couldn't find a suitable device");
    let physical_device = physical_devices[chosen_index];
    let physical_device_properties = physical_device_properties[chosen_index].clone();
    let physical_device_features = instance.get_physical_device_features(physical_device);

    (
        physical_device,
        physical_device_properties,
        physical_device_features,
        queue_families,
        selected_queue_families,
    )
}

impl GetContextStorage<object::Image> for InnerDevice {
    fn get_storage(ctx: &InnerDevice) -> &<object::Image as object::Object>::Storage {
        &ctx.image_storage
    }
}
