pub mod batch;
pub mod inflight;
pub mod submission;

use self::{
    batch::GenerationManager, inflight::InflightResourceManager, submission::SubmissionManager,
};
use super::instance::InstanceCreateInfo;
use crate::{
    instance::Instance,
    object::{self, Buffer, GraphicsPipeline, Image, Swapchain},
    storage::{nostore::SimpleStorage, ObjectStorage},
    tracing::shim_macros::{info, trace},
    util::format_utils::{self},
};
use pumice::{
    loader::{tables::DeviceTable, DeviceLoader},
    util::{ApiLoadConfig, ObjectHandle},
    vk,
    vk10::QueueFamilyProperties,
    DeviceWrapper, VulkanResult,
};
use pumice_vma::Allocator;
use smallvec::{smallvec, SmallVec};
use std::{
    collections::{hash_map::RandomState, HashSet},
    ffi::{c_void, CStr},
    fmt::Display,
    ops::Deref,
    ptr::NonNull,
    sync::Arc,
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
    pub instance: super::instance::OwnedInstance,
    pub config: &'a mut ApiLoadConfig<'a>,
    pub device_features: vk::PhysicalDeviceFeatures,
    pub queue_family_selection: &'a [QueueFamilySelection<'a>],
    /// substrings of device names, devices that contain them are prioritized
    pub device_substrings: &'a [&'a str],
    pub verbose: bool,
    // TODO verify that features are supported
    pub p_next: *const c_void,
}

pub struct Device {
    thread_pool: rayon::ThreadPool,
    pipeline_cache: vk::PipelineCache,

    pub(crate) device: pumice::DeviceWrapper,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) physical_device_properties: vk::PhysicalDeviceProperties,
    pub(crate) physical_device_features: vk::PhysicalDeviceFeatures,

    // family index, range for the subslice of the requested queues
    pub(crate) queue_selection_mapping: Vec<(usize, std::ops::Range<usize>)>,
    pub(crate) queues: Vec<vk::Queue>,
    pub(crate) queue_families: Vec<QueueFamilyProperties>,

    // object handle storage
    pub(crate) graphics_pipelines: SimpleStorage<GraphicsPipeline>,
    pub(crate) image_storage: SimpleStorage<Image>,
    pub(crate) buffer_storage: SimpleStorage<Buffer>,
    pub(crate) swapchain_storage: SimpleStorage<Swapchain>,

    // allocator
    pub(crate) allocator: Allocator,

    // synchronization
    pub(crate) synchronization_manager: parking_lot::RwLock<SubmissionManager>,
    // coarse grained synchronization
    pub(crate) generation_manager: parking_lot::RwLock<GenerationManager>,
    // tracks resources in flight that cannot be safely deleted
    pub(crate) pending_resources: parking_lot::RwLock<InflightResourceManager>,

    // at the bottom so that these are dropped last
    #[allow(unused)]
    pub(crate) device_table: Box<DeviceTable>,
    pub(crate) instance: super::instance::OwnedInstance,
}

#[derive(Clone)]
pub struct OwnedDevice(pub(crate) Arc<Device>);

impl Deref for OwnedDevice {
    type Target = Device;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl Device {
    pub unsafe fn new(info: DeviceCreateInfo) -> OwnedDevice {
        let DeviceCreateInfo {
            instance,
            config: conf,
            device_features,
            queue_family_selection,
            device_substrings,
            verbose: _,
            p_next,
        } = info;

        let allocation_callbacks = instance.allocator_callbacks();
        let instance_handle = instance.handle();

        // (_, _, _, overlay over queue_family_selection and their associated queue families)
        let (
            physical_device,
            physical_device_properties,
            physical_device_features,
            queue_families,
            selected_queue_families,
        ) = select_device(
            &instance.physical_devices,
            &instance.physical_device_properties,
            &instance_handle,
            &conf,
            &queue_family_selection,
            device_substrings,
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
            p_next,
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
            let info = pumice_vma::AllocatorCreateInfo2 {
                instance: instance.handle(),
                device: &device,
                physical_device: physical_device,
                flags: pumice_vma::AllocatorCreateFlags::empty(),
                preferred_large_heap_block_size: 0,
                allocation_callbacks,
                device_memory_callbacks: None,
                heap_size_limit: None,
                vulkan_api_version: conf.get_api_version(),
                external_memory_handle_types: None,
            };
            Allocator::new(&info).unwrap()
        };

        let name = instance.app_name().to_owned();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .thread_name(move |i| format!("{name} worker #{i}"))
            .build()
            .unwrap();

        let inner = Device {
            pipeline_cache: vk::PipelineCache::null(),
            thread_pool,

            instance,
            device,
            physical_device,
            physical_device_properties,
            physical_device_features,

            queue_families,
            queue_selection_mapping,
            queues,

            graphics_pipelines: SimpleStorage::new(),
            image_storage: SimpleStorage::new(),
            buffer_storage: SimpleStorage::new(),
            swapchain_storage: SimpleStorage::new(),

            allocator,

            synchronization_manager: parking_lot::RwLock::new(SubmissionManager::new()),
            generation_manager: parking_lot::RwLock::new(GenerationManager::new(10)),
            pending_resources: parking_lot::RwLock::new(InflightResourceManager::new()),

            device_table,
        };

        OwnedDevice(Arc::new(inner))
    }
    pub fn pipeline_cache(&self) -> vk::PipelineCache {
        self.pipeline_cache
    }
    pub fn device(&self) -> &pumice::DeviceWrapper {
        &self.device
    }
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }
    pub fn threadpool(&self) -> &rayon::ThreadPool {
        &self.thread_pool
    }
    pub fn allocator(&self) -> &Allocator {
        &self.allocator
    }
    pub fn allocator_callbacks(&self) -> Option<&vk::AllocationCallbacks> {
        self.instance.allocator_callbacks()
    }
    pub fn get_queue_bundle(
        &self,
        selection_index: usize,
        offset: usize,
    ) -> Option<submission::Queue> {
        Some(submission::Queue {
            raw: self.get_queue(selection_index, offset)?,
            family: self.get_queue_family(selection_index)?,
        })
    }
    pub fn get_queue(&self, selection_index: usize, offset: usize) -> Option<vk::Queue> {
        let range = self.queue_selection_mapping.get(selection_index)?.1.clone();
        self.queues.get(range)?.get(offset).cloned()
    }
    pub fn get_queue_family(&self, selection_index: usize) -> Option<u32> {
        self.queue_selection_mapping
            .get(selection_index)
            .map(|(family, _)| *family as u32)
    }
    pub fn get_queue_properties(
        &self,
        selection_index: usize,
    ) -> Option<&vk::QueueFamilyProperties> {
        self.queue_families
            .get(self.queue_selection_mapping.get(selection_index)?.0)
    }
    pub unsafe fn create_image(
        &self,
        info: object::ImageCreateInfo,
        allocate: pumice_vma::AllocationCreateInfo,
    ) -> VulkanResult<object::Image> {
        self.image_storage
            .get_or_create((info, allocate), NonNull::from(self))
            .map(object::Image)
    }
    pub unsafe fn create_swapchain(
        &self,
        info: object::SwapchainCreateInfo,
    ) -> VulkanResult<object::Swapchain> {
        self.swapchain_storage
            .get_or_create(info, NonNull::from(self))
            .map(object::Swapchain)
    }
    pub unsafe fn create_raw_semaphore(&self) -> VulkanResult<vk::Semaphore> {
        let info = vk::SemaphoreCreateInfo::default();
        self.device()
            .create_semaphore(&info, self.allocator_callbacks())
    }
    pub unsafe fn create_raw_fence(&self, signalled: bool) -> VulkanResult<vk::Fence> {
        let info = vk::FenceCreateInfo {
            flags: if signalled {
                vk::FenceCreateFlags::SIGNALED
            } else {
                vk::FenceCreateFlags::empty()
            },
            ..Default::default()
        };
        self.device()
            .create_fence(&info, self.allocator_callbacks())
    }
    pub unsafe fn destroy_raw_semaphore(&self, semaphore: vk::Semaphore) {
        self.device()
            .destroy_semaphore(semaphore, self.allocator_callbacks())
    }
    pub unsafe fn destroy_raw_fence(&self, fence: vk::Fence) {
        self.device()
            .destroy_fence(fence, self.allocator_callbacks())
    }
}

impl Drop for Device {
    // at this point, the reference count of the Arc is zero but it's possible that various
    // handles still have a NonNull<Device> in their headers
    //
    // there isn't much to be done, for those types of storage that can reach all their headers
    // can assert that all refcounts are zero, hovever this is not always possible, so we just tell
    // the users to not do that
    fn drop(&mut self) {
        unsafe {
            let callbacks = self.allocator_callbacks();

            self.device().device_wait_idle().unwrap();
            if self.pipeline_cache != vk::PipelineCache::null() {
                self.device()
                    .destroy_pipeline_cache(self.pipeline_cache, callbacks)
            }
            self.device().destroy_device(callbacks);
        }
    }
}

#[cfg(test)]
use {
    crate::tracing::tracing_subscriber::install_tracing_subscriber,
    crate::tracing::Severity,
    pumice_vma::{AllocationCreateFlags, AllocationCreateInfo},
};

#[test]
fn test_create_device() {
    unsafe {
        let _ = __test_init_device(false);
    }
}

#[test]
fn test_device() {
    install_tracing_subscriber(Some(Severity::Info));

    unsafe {
        let device = __test_init_device(false);

        let info = object::ImageCreateInfo {
            flags: vk::ImageCreateFlags::empty(),
            size: object::Extent::D2(1, 1),
            format: vk::Format::R8G8B8A8_SRGB,
            samples: vk::SampleCountFlags::C1,
            mip_levels: 1,
            array_layers: 1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            sharing_mode_concurrent: false,
            initial_layout: vk::ImageLayout::UNDEFINED,
        };
        let allocation_info = AllocationCreateInfo {
            flags: AllocationCreateFlags::MAPPED,
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
            preferred_flags: vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED,
            ..Default::default()
        };
        let image = device.create_image(info, allocation_info);

        // explicit drop order due to my crimes
        drop(image);
        drop(device);
    }
}

pub(crate) unsafe fn __test_init_device(mock_device: bool) -> OwnedDevice {
    let mut conf = ApiLoadConfig::new(vk::API_VERSION_1_0);
    let info = InstanceCreateInfo {
        config: &mut conf,
        validation_layers: &[pumice::cstr!("VK_LAYER_KHRONOS_validation")],
        enable_debug_callback: true,
        app_name: "test_context_new".to_owned(),
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
        // in case the user has the Mock ICD installed, possibly prefer that
        device_substrings: if mock_device { &["Mock"] } else { &[] },
        verbose: false,
        p_next: std::ptr::null(),
    };
    let device = Device::new(info);
    device
}

unsafe fn select_device(
    physical_devices: &Vec<vk::PhysicalDevice>,
    physical_device_properties: &Vec<vk::PhysicalDeviceProperties>,
    instance: &pumice::InstanceWrapper,
    conf: &ApiLoadConfig,
    queue_family_selection: &[QueueFamilySelection],
    device_substrings: &[&str],
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

    let mut device_considered: SmallVec<[bool; 16]> = smallvec![false; physical_devices.len()];
    let mut substrings = device_substrings.into_iter().peekable();

    // weird mimicry of an iterator
    let mut next = move || loop {
        // try to find a device that hasn't been considered that contains the requested substring
        if let Some(&&str) = substrings.peek() {
            let index = physical_device_properties
                .iter()
                .enumerate()
                .filter(|&(i, device)| {
                    device_considered[i] == false
                        &&
                        // FIXME quadratic complexity, maybe do this beforehand into a vector?
                        // though I think we can at most 4 devices in a system
                        CStr::from_ptr(device.device_name.as_ptr())
                            .to_str()
                            .expect("Device name is invalid UTF8")
                            .contains(str)
                })
                .next()
                .map(|(i, _)| i);

            if let Some(index) = index {
                device_considered[index] = true;
                return Some(index);
            } else {
                substrings.next();
                continue;
            }
        };

        // otherwise just return the first non-checked device
        let index = device_considered
            .iter()
            .enumerate()
            .filter(|&(_i, &checked)| checked == false)
            .next()
            .map(|(i, _)| i);

        if let Some(index) = index {
            device_considered[index] = true;
            return Some(index);
        } else {
            // we've exhausted all available devices
            return None;
        }
    };

    while let Some(i) = next() {
        let physical_device = physical_devices[i];
        let physical_device_properties = &physical_device_properties[i];

        let _device_name =
            CStr::from_ptr(physical_device_properties.device_name.as_ptr()).to_string_lossy();

        // extension criteria
        {
            let (extensions, _) = instance
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
                let _iter =
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

            for (_i, selection) in queue_family_selection.iter().enumerate() {
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
                    "Device '{}' skipped because it couldn't satisfy queue selection:\nTODO",
                    device_name, /* selection */
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
