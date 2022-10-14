use std::{
    borrow::Cow,
    collections::{hash_map::RandomState, HashSet},
    ffi::{c_char, CStr},
    fmt::{Display, Write},
    ops::Deref,
    str::FromStr,
};

use pumice::{
    loader::{
        tables::{DeviceTable, EntryTable, InstanceTable},
        InstanceLoader,
    },
    vk,
    vk10::QueueFamilyProperties,
    DeviceWrapper,
};
use tracing_subscriber::EnvFilter;

use crate::{
    install_tracing_subscriber,
    util::{
        self,
        debug_callback::to_version,
        format_utils::{self, Fun, IterDisplay},
    },
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

pub struct InnerContext {
    loader: pumice::loader::EntryLoader,
    tables: Box<(EntryTable, InstanceTable, DeviceTable)>,

    entry: pumice::EntryWrapper,
    instance: pumice::InstanceWrapper,
    device: pumice::DeviceWrapper,

    // physical_device_properties: vk::PhysicalDeviceProperties,
    allocator: pumice_vma::Allocator,
    allocation_callbacks: Option<vk::AllocationCallbacks>,

    physical_device_properties: vk::PhysicalDeviceProperties,
    physical_device_features: vk::PhysicalDeviceFeatures,

    queue_families: Vec<QueueFamilyProperties>,
    // family index, range for the subslice of the requested queues
    queue_selection_mapping: Vec<(usize, std::ops::Range<usize>)>,
    queues: Vec<vk::Queue>,

    debug_messenger: Option<pumice::extensions::ext_debug_utils::DebugUtilsMessengerEXT>,
    verbose: bool,
}

impl InnerContext {
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
            tracing::info!(
                "{} instance layers:\n{}",
                instance_layer_properties.len(),
                iter
            );
        } else {
            tracing::info!("{} instance layers", instance_layer_properties.len());
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
            tracing::info!("{} physical devices:{}", physical_devices.len(), fun);
        } else {
            tracing::info!("{} physical devices", physical_devices.len());
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

        Self {
            loader,
            tables,

            entry,
            instance,
            device,

            allocator,
            allocation_callbacks,

            physical_device_properties,
            physical_device_features,

            queue_families,
            queue_selection_mapping,
            queues,

            debug_messenger,
            verbose,
        }
    }
    pub fn entry(&self) -> &pumice::EntryWrapper {
        &self.entry
    }
    pub fn instance(&self) -> &pumice::InstanceWrapper {
        &self.instance
    }
    pub fn device(&self) -> &pumice::DeviceWrapper {
        &self.device
    }
    pub fn allocator_callbacks(&self) -> Option<&vk::AllocationCallbacks> {
        self.allocation_callbacks.as_ref()
    }
    pub fn get_queue(&self, selection_index: usize, offset: usize) -> Option<vk::Queue> {
        let range = self.queue_selection_mapping.get(selection_index)?.1.clone();
        self.queues.get(range)?.get(offset).cloned()
    }
    pub fn get_queue_family(&self, selection_index: usize, offset: usize) -> Option<u32> {
        self.queue_selection_mapping
            .get(selection_index)
            .map(|(family, _)| *family as u32)
    }
}

#[test]
fn test_context_new() {
    install_tracing_subscriber(Some(EnvFilter::from_str("TRACE").unwrap()));

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
            coalesce: true,
        }],
        app_name: pumice::cstr!("test_context_new"),
    };

    unsafe {
        let context = InnerContext::new(info);
    }
}

// impl Deref for InnerContext {
//     type Target = pumice::DeviceWrapper;
//     fn deref(&self) -> &Self::Target {
//         &self.device
//     }
// }

unsafe fn select_device(
    physical_devices: &Vec<pumice::vk10::PhysicalDevice>,
    physical_device_properties: &Vec<pumice::vk10::PhysicalDeviceProperties>,
    instance: &pumice::InstanceWrapper,
    conf: &pumice::util::config::ApiLoadConfig,
    queue_family_selection: &Vec<QueueFamilySelection>,
) -> (
    pumice::vk10::PhysicalDevice,
    pumice::vk10::PhysicalDeviceProperties,
    pumice::vk10::PhysicalDeviceFeatures,
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

    'devices: for (i, (physical_device, physical_device_properties)) in physical_devices
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

                tracing::trace!("Device '{device_name}' is missing extensions:\n{iter}",);
                tracing::info!("Device '{}' skipped due to missing extensions", device_name);

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

                // first try to coalesce
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

                tracing::info!(
                    "Device '{}' skipped because it couldn't satisfy queue selection:\n{:?}",
                    device_name,
                    selection
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

        tracing::info!("Selected device '{}'", device_name);
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
