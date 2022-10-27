use super::instance::{Instance, InstanceCreateInfo};
use crate::{
    object::{self},
    storage::{nostore::NoStore, ObjectStorage},
    synchronization::InnerSynchronizationManager,
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
        DeviceLoader, InstanceLoader,
    },
    util::{config::ApiLoadConfig, result::VulkanResult},
    vk,
    vk10::QueueFamilyProperties,
    DeviceWrapper,
};
use pumice_vma::{Allocator, AllocatorCreateInfo};
use std::{
    collections::{hash_map::RandomState, HashSet},
    ffi::CStr,
    fmt::Display,
    pin::Pin,
    sync::Mutex,
};

#[derive(Debug, Clone, Copy)]
pub enum SelectionMechanism {
    Exact,
    Contains,
}

#[derive(Clone)]
pub struct QueueFamilySelection<'a> {
    pub mask: vk::QueueFlags,
    pub count: u32,
    pub priority: f32,
    pub exact: bool,
    pub attempt_dedicated: bool,
    pub coalesce: bool,
    pub support_surfaces: &'a [&'a object::Surface],
}

pub struct DeviceCreateInfo<'a> {
    pub instance: super::instance::Instance,
    pub config: &'a mut ApiLoadConfig<'a>,
    pub device_features: vk::PhysicalDeviceFeatures,
    pub queue_family_selection: &'a [QueueFamilySelection<'a>],
    pub verbose: bool,
}

pub(crate) struct InnerDevice {
    pub(crate) device: pumice::DeviceWrapper,
    pub(crate) physical_device_properties: vk::PhysicalDeviceProperties,
    pub(crate) physical_device_features: vk::PhysicalDeviceFeatures,

    // family index, range for the subslice of the requested queues
    pub(crate) queue_selection_mapping: Vec<(usize, std::ops::Range<usize>)>,
    pub(crate) queues: Vec<vk::Queue>,
    pub(crate) queue_families: Vec<QueueFamilyProperties>,

    // object handle storage
    pub(crate) image_storage: NoStore,
    pub(crate) buffer_storage: NoStore,
    pub(crate) swapchain_storage: NoStore,

    // allocator
    pub(crate) allocator: Allocator,

    // synchronization
    pub(crate) synchronization_manager: parking_lot::RwLock<InnerSynchronizationManager>,

    // at the bottom so that these are dropped last
    pub(crate) device_table: Box<DeviceTable>,
    pub(crate) instance: super::instance::Instance,
}

pub struct Device(pub(crate) Pin<Box<InnerDevice>>);

impl Device {
    pub unsafe fn new(info: DeviceCreateInfo) -> Self {
        let DeviceCreateInfo {
            instance,
            config: conf,
            device_features,
            queue_family_selection,
            verbose,
        } = info;

        let allocation_callbacks = instance.allocator_callbacks();
        let inner = instance.inner();
        let instance_handle = instance.handle();

        // (_, _, _, overlay over queue_family_selection and their associated queue families)
        let (
            physical_device,
            physical_device_properties,
            physical_device_features,
            queue_families,
            selected_queue_families,
        ) = select_device(
            &inner.physical_devices,
            &inner.physical_device_properties,
            &instance_handle,
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

        let device_handle = instance_handle
            .create_device(physical_device, &device_create_info, allocation_callbacks)
            .expect("create_device error");

        let (device, device_table) = {
            let mut table = Box::new(DeviceTable::new_empty());
            let loader = DeviceLoader::new(device_handle, &instance.instance_loader());
            table.load(&loader, conf);
            let device = DeviceWrapper::new(device_handle, &*table);

            (device, table)
        };

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
            let create_info = AllocatorCreateInfo::new(&instance_handle, &device, &physical_device);
            Allocator::new(create_info).unwrap()
        };

        let inner = InnerDevice {
            instance,
            device,
            physical_device_properties,
            physical_device_features,

            queue_families,
            queue_selection_mapping,
            queues,

            image_storage: NoStore::new(),
            buffer_storage: NoStore::new(),
            swapchain_storage: NoStore::new(),

            allocator,

            synchronization_manager: parking_lot::RwLock::new(InnerSynchronizationManager::new()),

            device_table,
        };

        Self(Box::pin(inner))
    }
    pub fn device(&self) -> &pumice::DeviceWrapper {
        &self.0.device
    }
    pub fn allocator(&self) -> &Allocator {
        &self.0.allocator
    }
    pub fn allocator_callbacks(&self) -> Option<&vk::AllocationCallbacks> {
        self.0.instance.allocator_callbacks()
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
            .get_or_create(info, allocate, &*self.0)
            .map(object::Image)
    }

    pub unsafe fn create_swapchain(
        &self,
        info: object::SwapchainCreateInfo,
    ) -> VulkanResult<object::Swapchain> {
        self.0
            .swapchain_storage
            .get_or_create(info, (), &*self.0)
            .map(object::Swapchain)
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
fn test_device() {
    install_tracing_subscriber(Severity::Info);

    unsafe {
        let mut conf = ApiLoadConfig::new(vk::API_VERSION_1_0);

        let info = InstanceCreateInfo {
            config: &mut conf,
            validation_layers: &[pumice::cstr!("VK_LAYER_KHRONOS_validation")],
            enable_debug_callback: true,
            app_name: pumice::cstr!("test_context_new"),
            verbose: false,
        };

        let instance = Instance::new(info);

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
                support_surfaces: &[],
            }],
            verbose: false,
        };

        let device = Device::new(info);

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
        let image = device.create_image(info, allocation_info);

        // explicit drop order due to my crimes
        drop(image);
        drop(device);
    }
}

unsafe fn select_device(
    physical_devices: &Vec<vk::PhysicalDevice>,
    physical_device_properties: &Vec<vk::PhysicalDeviceProperties>,
    instance: &pumice::InstanceWrapper,
    conf: &ApiLoadConfig,
    queue_family_selection: &[QueueFamilySelection],
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

                let surface_supported = |queue_family_index: usize| {
                    selection.support_surfaces.iter().all(|s| {
                        instance
                            .get_physical_device_surface_support_khr(
                                physical_device,
                                queue_family_index as u32,
                                s.handle(),
                            )
                            .unwrap()
                            != vk::FALSE
                    })
                };

                let family_iter = queue_families
                    .iter()
                    .zip(&family_search_scratch)
                    .enumerate();

                // try to select some family that does not have queues
                if selection.attempt_dedicated {
                    let position = family_iter
                        .clone()
                        .position(|(i, (family, selected_count))| {
                            *selected_count == 0
                                && is_valid(family.queue_flags, selection.mask)
                                && *selected_count + selection.count <= family.queue_count
                                && surface_supported(i)
                        });

                    if let Some(position) = position {
                        family_search_scratch[position] += selection.count;
                        selected_queue_families.push(position);
                        continue;
                    }
                }
                // try to select some family that already has queues
                if selection.coalesce {
                    let position = family_iter
                        .clone()
                        .position(|(i, (family, selected_count))| {
                            *selected_count > 0
                                && is_valid(family.queue_flags, selection.mask)
                                && *selected_count + selection.count <= family.queue_count
                                && surface_supported(i)
                        });

                    if let Some(position) = position {
                        family_search_scratch[position] += selection.count;
                        selected_queue_families.push(position);
                        continue;
                    }
                }
                // then try to match anything
                {
                    let position = family_iter
                        .clone()
                        .position(|(i, (family, selected_count))| {
                            is_valid(family.queue_flags, selection.mask)
                                && *selected_count + selection.count <= family.queue_count
                                && surface_supported(i)
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
