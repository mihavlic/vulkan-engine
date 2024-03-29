pub mod batch;
pub mod debug;
pub mod reflection;
pub mod ring;
pub mod staging;
pub mod submission;

pub use {batch::*, debug::*, reflection::*, ring::*, staging::*, submission::*};

use self::{batch::GenerationManager, staging::StagingManager, submission::SubmissionManager};
use super::instance::InstanceCreateInfo;
use crate::{
    instance::{self, Instance},
    object::{
        self, create_compute_pipeline_impl, Buffer, ComputePipeline, DescriptorSetLayout,
        Framebuffer, GraphicsPipeline, Image, PipelineLayout, RenderPass, RenderPassMode, Sampler,
        ShaderModule, Swapchain,
    },
    storage::{nostore::SimpleStorage, ObjectStorage},
    tracing::shim_macros::{info, trace},
    util::format_utils::{self},
};
use bumpalo::Bump;
use pumice::{
    loader::{tables::DeviceTable, DeviceLoader},
    util::{ApiLoadConfig, ObjectHandle},
    vk,
    vk10::QueueFamilyProperties,
    DeviceWrapper, VulkanResult,
};
use pumice_vma::Allocator;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use smallvec::{smallvec, SmallVec};
use spirq::Variable;
use std::{
    collections::{hash_map::RandomState, HashSet},
    ffi::{c_void, CStr},
    fmt::Display,
    io,
    mem::ManuallyDrop,
    ops::Deref,
    ptr::NonNull,
    slice,
    sync::{atomic::AtomicU64, Arc},
    time::Duration,
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
    pub staging_transfer_queue: (usize, usize),
    /// substrings of device names, devices that contain them are prioritized
    pub device_substrings: &'a [&'a str],
    pub verbose: bool,
    // TODO verify that features are supported
    pub p_next: *const c_void,
}

pub struct Device {
    threadpool: ManuallyDrop<rayon::ThreadPool>,
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
    pub(crate) compute_pipelines: SimpleStorage<ComputePipeline>,
    pub(crate) shader_modules: SimpleStorage<ShaderModule>,
    pub(crate) samplers: SimpleStorage<Sampler>,
    pub(crate) framebuffers: SimpleStorage<Framebuffer>,
    pub(crate) render_passes: SimpleStorage<RenderPass>,
    pub(crate) pipeline_layouts: SimpleStorage<PipelineLayout>,
    pub(crate) descriptor_set_layouts: SimpleStorage<DescriptorSetLayout>,
    pub(crate) image_storage: SimpleStorage<Image>,
    pub(crate) buffer_storage: SimpleStorage<Buffer>,
    pub(crate) swapchain_storage: SimpleStorage<Swapchain>,

    // allocator
    pub(crate) allocator: Allocator,

    // staged memory transfers to and fro
    pub(crate) staging_manager: parking_lot::RwLock<StagingManager>,
    // synchronization
    pub(crate) synchronization_manager: parking_lot::RwLock<SubmissionManager>,
    // coarse grained synchronization
    pub(crate) generation_manager: parking_lot::RwLock<GenerationManager>,

    // at the bottom so that these are dropped last
    #[allow(unused)]
    pub(crate) device_table: Box<DeviceTable>,
    pub(crate) instance: super::instance::OwnedInstance,
}

#[derive(Clone)]
pub struct OwnedDevice(pub(crate) Arc<Device>);

impl OwnedDevice {
    pub unsafe fn attempt_destroy(self) -> Result<(), ()> {
        match Arc::try_unwrap(self.0) {
            Ok(unique) => {
                drop(unique);
                Ok(())
            }
            Err(returned) => {
                drop(returned);
                Err(())
            }
        }
    }
}

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
            staging_transfer_queue,
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

        let threadpool = rayon::ThreadPoolBuilder::new()
            .thread_name(move |i| format!("worker #{i}"))
            .build()
            .unwrap();

        let staging_transfer_queue = {
            let (index, offset) = staging_transfer_queue;
            let (family, range) = queue_selection_mapping[index].clone();
            let queue = queues[range][offset];

            submission::Queue {
                raw: queue,
                family: family as u32,
            }
        };

        let staging_manager = StagingManager::new(
            staging_transfer_queue,
            instance.allocator_callbacks(),
            &device,
        );

        let inner = Device {
            threadpool: ManuallyDrop::new(threadpool),
            pipeline_cache: vk::PipelineCache::null(),

            device,
            physical_device,
            physical_device_properties,
            physical_device_features,
            queue_selection_mapping,

            queues,
            queue_families,
            graphics_pipelines: SimpleStorage::new(),

            compute_pipelines: SimpleStorage::new(),
            shader_modules: SimpleStorage::new(),
            samplers: SimpleStorage::new(),
            framebuffers: SimpleStorage::new(),
            render_passes: SimpleStorage::new(),
            pipeline_layouts: SimpleStorage::new(),
            descriptor_set_layouts: SimpleStorage::new(),
            image_storage: SimpleStorage::new(),
            buffer_storage: SimpleStorage::new(),
            swapchain_storage: SimpleStorage::new(),
            allocator,

            staging_manager: parking_lot::RwLock::new(staging_manager),
            synchronization_manager: parking_lot::RwLock::new(SubmissionManager::new()),
            generation_manager: parking_lot::RwLock::new(GenerationManager::new(10)),
            device_table,

            instance,
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
        &self.threadpool
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
    pub fn find_queue_for_family(&self, queue_family: u32) -> Option<vk::Queue> {
        let range = self
            .queue_selection_mapping
            .iter()
            .find(|&&(family, _)| family == queue_family as usize)?
            .1
            .clone();
        self.queues.get(range)?.get(0).cloned()
    }
    pub unsafe fn create_shader_module_read<R: io::Read + io::Seek>(
        &self,
        spirv: &mut R,
    ) -> VulkanResult<object::ShaderModule> {
        let data = read_spirv(spirv).map_err(|_| vk::Result::ERROR_UNKNOWN)?;
        self.shader_modules
            .get_or_create(data.as_ref(), self)
            .map(object::ShaderModule)
    }
    pub unsafe fn create_shader_module_spirv_unaligned(
        &self,
        spirv: impl AsRef<[u8]>,
    ) -> VulkanResult<object::ShaderModule> {
        let mut cursor = std::io::Cursor::new(spirv.as_ref());
        self.create_shader_module_read(&mut cursor)
    }
    pub unsafe fn create_shader_module_spirv(
        &self,
        spirv: impl AsRef<[u32]>,
    ) -> VulkanResult<object::ShaderModule> {
        self.shader_modules
            .get_or_create(spirv.as_ref(), self)
            .map(object::ShaderModule)
    }
    pub unsafe fn create_descriptor_sampler(
        &self,
        info: object::SamplerCreateInfo,
    ) -> VulkanResult<object::Sampler> {
        self.samplers.get_or_create(info, self).map(object::Sampler)
    }
    pub unsafe fn create_descriptor_set_layout(
        &self,
        info: object::DescriptorSetLayoutCreateInfo,
    ) -> VulkanResult<object::DescriptorSetLayout> {
        self.descriptor_set_layouts
            .get_or_create(info, self)
            .map(object::DescriptorSetLayout)
    }
    pub unsafe fn create_render_pass(
        &self,
        info: object::RenderPassCreateInfo,
    ) -> VulkanResult<object::RenderPass> {
        self.render_passes
            .get_or_create(info, self)
            .map(object::RenderPass)
    }
    pub unsafe fn create_framebuffer(
        &self,
        info: object::FramebufferCreateInfo,
    ) -> VulkanResult<object::Framebuffer> {
        self.framebuffers
            .get_or_create(info, self)
            .map(object::Framebuffer)
    }
    pub unsafe fn create_pipeline_layout(
        &self,
        info: object::PipelineLayoutCreateInfo,
    ) -> VulkanResult<object::PipelineLayout> {
        self.pipeline_layouts
            .get_or_create(info, self)
            .map(object::PipelineLayout)
    }
    pub unsafe fn create_delayed_graphics_pipeline(
        &self,
        info: object::GraphicsPipelineCreateInfo,
    ) -> object::GraphicsPipeline {
        assert!(matches!(info.render_pass, RenderPassMode::Delayed));
        self.graphics_pipelines
            .get_or_create(info, self)
            .map(object::GraphicsPipeline)
            // infallible since the pipeline creation is delayed and no api calls are made
            .unwrap()
    }
    // FIXME this currently uses the same type as delayed pipelines which may be more elegant
    // but in cases where we know the pipeline's formats we don't really need all the machinery
    pub unsafe fn create_graphics_pipeline(
        &self,
        info: object::GraphicsPipelineCreateInfo,
    ) -> VulkanResult<object::ConcreteGraphicsPipeline> {
        let mut info = info;
        assert!(
            !matches!(info.render_pass, RenderPassMode::Delayed),
            "A delayed pipeline cannot be created as concrete"
        );
        // eh
        let mode = std::mem::take(&mut info.render_pass);

        let delayed = self
            .graphics_pipelines
            .get_or_create(info, self)
            .map(object::GraphicsPipeline)
            // infallible since the pipeline creation is delayed and no api calls are made
            .unwrap();

        delayed.get_concrete_for_mode(mode, self)
    }
    pub unsafe fn create_compute_pipeline(
        &self,
        info: object::ComputePipelineCreateInfo,
    ) -> VulkanResult<object::ComputePipeline> {
        self.compute_pipelines
            .get_or_create(info, self)
            .map(object::ComputePipeline)
    }
    pub unsafe fn create_compute_pipelines_parallel(
        &self,
        infos: Vec<object::ComputePipelineCreateInfo>,
    ) -> VulkanResult<Vec<object::ComputePipeline>> {
        let datas = self.threadpool().install(|| {
            infos
                .into_par_iter()
                .map(|info| {
                    // FIXME Bump is not Sync
                    let bump = Bump::new();
                    create_compute_pipeline_impl(info, &bump, self)
                })
                .collect::<VulkanResult<Vec<_>>>()
        })?;

        let mut pipelines = self.compute_pipelines.add_multiple(datas, self);
        // FIXME use Vec::into_raw_parts
        let (ptr, len, cap) = (
            pipelines.as_mut_ptr(),
            pipelines.len(),
            pipelines.capacity(),
        );
        std::mem::forget(pipelines);

        // we're converting a Vec<ObjHandle<ComputePipeline>> into a Vec<ComputeHandle>
        // since ComputeHandle is repr(transparent)
        Ok(Vec::from_raw_parts(ptr.cast(), len, cap))
    }
    pub unsafe fn create_image(
        &self,
        info: object::ImageCreateInfo,
        allocate: pumice_vma::AllocationCreateInfo,
    ) -> VulkanResult<object::Image> {
        self.image_storage
            .get_or_create((info, allocate), self)
            .map(object::Image)
    }
    pub unsafe fn create_buffer(
        &self,
        info: object::BufferCreateInfo,
        allocate: pumice_vma::AllocationCreateInfo,
    ) -> VulkanResult<object::Buffer> {
        self.buffer_storage
            .get_or_create((info, allocate), self)
            .map(object::Buffer)
    }
    pub unsafe fn create_swapchain(
        &self,
        info: object::SwapchainCreateInfo,
    ) -> VulkanResult<object::Swapchain> {
        self.swapchain_storage
            .get_or_create(info, self)
            .map(object::Swapchain)
    }
    pub unsafe fn create_raw_semaphore(&self) -> VulkanResult<vk::Semaphore> {
        let info = vk::SemaphoreCreateInfo::default();
        self.device()
            .create_semaphore(&info, self.allocator_callbacks())
    }
    pub unsafe fn create_raw_timeline_semaphore(&self) -> VulkanResult<vk::Semaphore> {
        let p_next = vk::SemaphoreTypeCreateInfoKHR {
            semaphore_type: vk::SemaphoreTypeKHR::TIMELINE,
            initial_value: 0,
            ..Default::default()
        };
        let info = vk::SemaphoreCreateInfo {
            p_next: &p_next as *const _ as *const c_void,
            ..Default::default()
        };

        self.device()
            .create_semaphore(&info, self.allocator_callbacks())
    }
    pub unsafe fn create_raw_fence(&self, signaled: bool) -> VulkanResult<vk::Fence> {
        let info = vk::FenceCreateInfo {
            flags: if signaled {
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
    pub fn instance(&self) -> &instance::OwnedInstance {
        &self.instance
    }
    pub fn debug(&self) -> bool {
        self.instance.debug()
    }
    pub fn drain_work(&self) {
        self.wait_idle();
        let mut synchronization_manager = self.synchronization_manager.write();
        let mut generation_manager = self.generation_manager.write();
        synchronization_manager.wait_all(&self);
        unsafe { generation_manager.clear() };
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
            let threadpool = ManuallyDrop::take(&mut self.threadpool);
            drop(threadpool);

            self.drain_work();

            let callbacks = self.allocator_callbacks();

            self.synchronization_manager.write().destroy(self);
            self.generation_manager.write().destroy(self);
            self.staging_manager.write().destroy(self);

            if self.pipeline_cache != vk::PipelineCache::null() {
                self.device()
                    .destroy_pipeline_cache(self.pipeline_cache, callbacks);
            }

            self.graphics_pipelines.cleanup();
            self.compute_pipelines.cleanup();

            self.pipeline_layouts.cleanup();
            self.descriptor_set_layouts.cleanup();

            self.shader_modules.cleanup();
            self.samplers.cleanup();

            self.image_storage.cleanup();
            self.buffer_storage.cleanup();
            self.swapchain_storage.cleanup();

            self.render_passes.cleanup();

            pumice_vma::vmaDestroyAllocator(self.allocator.clone());

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
            ..Default::default()
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
        debug_labeling: true,
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
        staging_transfer_queue: (0, 0),
        // in case the user has the Mock ICD installed, possibly prefer that
        device_substrings: if mock_device { &["Mock"] } else { &[] },
        verbose: false,
        p_next: std::ptr::null(),
    };
    let device = Device::new(info);
    device
}

#[allow(unused)]
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

        let device_name =
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
                    "Device '{device_name}' skipped because it couldn't satisfy queue selection:\nTODO",
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

// stolen from ash
pub fn read_spirv<R: io::Read + io::Seek>(x: &mut R) -> io::Result<Vec<u32>> {
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
        std::slice::from_raw_parts_mut(result.as_mut_ptr().cast::<u8>(), words * 4)
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
