use core::panic;
use std::{
    backtrace::Backtrace,
    borrow::{BorrowMut, Cow},
    cell::{Cell, Ref, RefCell, RefMut},
    collections::{hash_map::Entry, BinaryHeap},
    fmt::Display,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut, Range},
    sync::Arc,
};

use ahash::HashMap;
use bumpalo::Bump;
use fixedbitset::FixedBitSet;
use pumice::{util::ObjectHandle, DeviceWrapper};
use rayon::ThreadPool;
use slice_group_by::GroupBy;
use smallvec::{smallvec, SmallVec};

use crate::{
    arena::uint::{Config, OptionalU32, PackedUint},
    device::{
        batch::GenerationId,
        submission::{self, QueueSubmission, SubmissionData, TimelineSemaphore},
        OwnedDevice,
    },
    graph::{
        allocator::MemoryKind,
        compile::CombinedResourceHandle,
        resource_marker::{BufferMarker, ImageMarker},
        reverse_edges::reverse_edges_into,
        task::GraphicsPipelineSrc,
    },
    object::{
        self, raw_info_handle_renderpass, ConcreteGraphicsPipeline, GetPipelineResult,
        GraphicsPipeline, ObjRef, RenderPassMode, SwapchainAcquireStatus, SynchronizationState,
        SynchronizeResult,
    },
    passes::RenderPass,
    simple_handle,
    storage::{constant_ahash_hashmap, constant_ahash_hashset, ObjectStorage},
    util::ffi_ptr::AsFFiPtr,
};

use super::{
    allocator::{AvailabilityToken, Suballocator},
    compile::{
        BufferBarrier, GraphPassEvent, GraphResource, ImageBarrier, ImageKindCreateInfo,
        MemoryBarrier, PassMeta, PassObjectState, ResourceFirstAccess, ResourceState,
        ResourceSubresource, Submission, SwapchainPresent,
    },
    descriptors::{
        bind_descriptor_sets, DescriptorAllocator, FinishedSet, UniformMemory, UniformResult,
    },
    record::{BufferData, CompilationInput, GraphBuilder, ImageData, ImageMove, PassData},
    resource_marker::{ResourceMarker, TypeNone, TypeOption, TypeSome},
    reverse_edges::{ChildRelativeKey, DFSCommand, ImmutableGraph, NodeGraph, NodeKey},
    task::{
        CompileGraphicsPipelinesTask, ComputePipelinePromise, ExecuteFnTask,
        GraphicsPipelinePromise, GraphicsPipelineResult, Promise, SendAny, SendUnsafeCell,
    },
    GraphBuffer, GraphImage, GraphPass, GraphPassMove, GraphQueue, GraphSubmission,
    ObjectSafeCreatePass, PhysicalBuffer, PhysicalImage, RawHandle, SubmissionPass, TimelinePass,
};

use pumice::vk;

use crate::{
    device::Device,
    object::{BufferCreateInfo, BufferMutableState, ImageCreateInfo, ImageMutableState},
};

use super::allocator::SuballocationUgh;

pub(crate) struct PhysicalImageData {
    pub(crate) info: ImageCreateInfo,
    // pub(crate) memory: SuballocationUgh,
    pub(crate) vkhandle: vk::Image,
    pub(crate) state: RefCell<ImageMutableState>,
}

pub(crate) struct PhysicalBufferData {
    pub(crate) info: BufferCreateInfo,
    // pub(crate) memory: SuballocationUgh,
    pub(crate) vkhandle: vk::Buffer,
    pub(crate) state: RefCell<BufferMutableState>,
}

#[derive(Default)]
pub(crate) struct LegacySemaphoreStack {
    next_index: usize,
    semaphores: Vec<vk::Semaphore>,
}

impl LegacySemaphoreStack {
    pub(crate) fn new() -> Self {
        Self::default()
    }
    pub(crate) unsafe fn reset(&mut self) {
        self.next_index = 0;
    }
    pub(crate) unsafe fn next(&mut self, ctx: &Device) -> vk::Semaphore {
        if self.next_index == self.semaphores.len() {
            let semaphore = ctx.create_raw_semaphore().unwrap();
            self.semaphores.push(semaphore);
        }

        let semaphore = self.semaphores[self.next_index];
        self.next_index += 1;
        semaphore
    }
    pub(crate) unsafe fn destroy(&mut self, ctx: &Device) {
        for semaphore in self.semaphores.drain(..) {
            ctx.destroy_raw_semaphore(semaphore);
        }
    }
}

impl Drop for LegacySemaphoreStack {
    fn drop(&mut self) {
        if !self.semaphores.is_empty() {
            panic!("LegacySemaphoreStack has not been destroyed before being dropped!");
        }
    }
}

struct CommandBufferPool {
    queue_family: u32,
    pool: vk::CommandPool,
    next_index: usize,
    buffers: Vec<vk::CommandBuffer>,
}

impl CommandBufferPool {
    unsafe fn new(queue_family: u32, device: &Device) -> Self {
        let info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::TRANSIENT,
            queue_family_index: queue_family,
            ..Default::default()
        };

        Self {
            queue_family,
            pool: device
                .device()
                .create_command_pool(&info, device.allocator_callbacks())
                .unwrap(),
            next_index: 0,
            buffers: Vec::new(),
        }
    }
    unsafe fn reset(&mut self, device: &Device) {
        self.next_index = 0;
        device.device().reset_command_pool(self.pool, None).unwrap();
    }
    unsafe fn destroy(&mut self, device: &Device) {
        device
            .device()
            .destroy_command_pool(self.pool, device.allocator_callbacks());
    }
    unsafe fn next(&mut self, device: &Device) -> vk::CommandBuffer {
        if self.next_index == self.buffers.len() {
            let info = vk::CommandBufferAllocateInfo {
                command_pool: self.pool,
                level: vk::CommandBufferLevel::PRIMARY,
                // possible allocate more at a time
                command_buffer_count: 1,
                ..Default::default()
            };
            let allocated = device.device().allocate_command_buffers(&info).unwrap();
            self.buffers.extend(allocated);
        }

        let buffer = self.buffers[self.next_index];
        self.next_index += 1;
        buffer
    }
}

pub(crate) struct CommandBufferStack {
    families: Vec<CommandBufferPool>,
}

impl CommandBufferStack {
    pub(crate) fn new() -> Self {
        Self {
            families: Vec::new(),
        }
    }
    pub(crate) unsafe fn reset(&mut self, device: &Device) {
        for family in &mut self.families {
            family.reset(device);
        }
    }
    pub(crate) unsafe fn next(&mut self, queue_family: u32, device: &Device) -> vk::CommandBuffer {
        let buffer = match self
            .families
            .binary_search_by_key(&queue_family, |f| f.queue_family)
        {
            Ok(ok) => &mut self.families[ok],
            Err(insert) => {
                let new = CommandBufferPool::new(queue_family, device);
                self.families.insert(insert, new);
                &mut self.families[insert]
            }
        };

        buffer.next(device)
    }
    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
        for family in &mut self.families {
            family.destroy(device);
        }
    }
}

pub struct GraphExecutor<'a> {
    pub(crate) graph: &'a CompiledGraph,
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) swapchain_image_indices: &'a ahash::HashMap<GraphImage, u32>,
    pub(crate) raw_images: &'a [Option<vk::Image>],
    pub(crate) raw_buffers: &'a [Option<vk::Buffer>],
}

impl<'a> GraphExecutor<'a> {
    pub fn get_image_format(&self, image: GraphImage) -> vk::Format {
        let info = self.graph.get_image_create_info(image);
        match info {
            ImageKindCreateInfo::Image(info) => info.format,
            ImageKindCreateInfo::ImageRef(info) => info.format,
            ImageKindCreateInfo::Swapchain(info) => info.format,
        }
    }
    pub fn get_image_extent(&self, image: GraphImage) -> object::Extent {
        let info = self.graph.get_image_create_info(image);
        match info {
            ImageKindCreateInfo::Image(info) => info.size,
            ImageKindCreateInfo::ImageRef(info) => info.size,
            ImageKindCreateInfo::Swapchain(info) => {
                let ImageData::Swapchain(archandle) = self.graph.input.get_image_data(image) else {
                    unreachable!()
                };
                let extent = archandle.get_extent();
                object::Extent::D2(extent.width, extent.height)
            }
        }
    }
    pub fn get_image_subresource_range(
        &self,
        image: GraphImage,
        aspect: vk::ImageAspectFlags,
    ) -> vk::ImageSubresourceRange {
        self.graph.input.get_image_subresource_range(image, aspect)
    }
    pub unsafe fn get_image_view(
        &self,
        image: GraphImage,
        info: &object::ImageViewCreateInfo,
    ) -> vk::ImageView {
        assert!(
            self.graph.is_image_alive(image),
            "Attempt to access dead image"
        );

        let mut info = info.clone();
        let layer_offset = self.graph.input.get_image_subresource_layer_offset(image);
        info.subresource_range.base_array_layer += layer_offset;

        let batch_id = self.graph.current_generation.get().unwrap();

        // TODO implement some sort of cache for views, since we will be using GenerationId's to garbage collect views
        // we can rely on them being present for our entire generation and don't need to keep locking the handle
        let data = self.graph.input.get_concrete_image_data(image);
        match data {
            ImageData::TransientPrototype(_, _) => {
                panic!("A compiled graph cannot have prototype transient resources")
            }
            ImageData::Transient(physical) => {
                let data = self.graph.get_physical_image(*physical);
                let view = data
                    .state
                    .borrow_mut()
                    .get_view(data.vkhandle, &info, batch_id, &self.graph.state().device)
                    .unwrap();
                view
            }
            ImageData::Imported(archandle) => archandle.get_view(&info, batch_id).unwrap(),
            ImageData::Swapchain(archandle) => {
                let index = *self.swapchain_image_indices.get(&image).unwrap();
                archandle.get_view(index, &info, batch_id).unwrap()
            }
            ImageData::Moved(_) => {
                panic!("get_concrete_image_data should resolve moved images into their targets")
            }
        }
    }
    pub unsafe fn get_default_image_view(
        &self,
        image: GraphImage,
        aspect: vk::ImageAspectFlags,
    ) -> vk::ImageView {
        let info = self.graph.get_image_create_info(image);
        let (kind, format) = match info {
            ImageKindCreateInfo::Image(info) => (
                match info.size {
                    object::Extent::D1(_) => vk::ImageViewType::T1D,
                    object::Extent::D2(_, _) => vk::ImageViewType::T2D,
                    object::Extent::D3(_, _, _) => vk::ImageViewType::T3D,
                },
                info.format,
            ),
            ImageKindCreateInfo::ImageRef(info) => (
                match info.size {
                    object::Extent::D1(_) => vk::ImageViewType::T1D,
                    object::Extent::D2(_, _) => vk::ImageViewType::T2D,
                    object::Extent::D3(_, _, _) => vk::ImageViewType::T3D,
                },
                info.format,
            ),
            ImageKindCreateInfo::Swapchain(info) => (vk::ImageViewType::T2D, info.format),
        };
        let subresource_range = self.get_image_subresource_range(image, aspect);
        self.get_image_view(
            image,
            &object::ImageViewCreateInfo {
                view_type: kind,
                format,
                components: vk::ComponentMapping::default(),
                subresource_range,
            },
        )
    }
    pub fn get_image(&self, image: GraphImage) -> vk::Image {
        self.raw_images[image.index()].unwrap_or_else(|| {
            panic!(
                "Missing handle for '{}', is the image dead?",
                self.graph.input.get_image_display(image)
            )
        })
    }
    pub fn get_buffer(&self, buffer: GraphBuffer) -> vk::Buffer {
        self.raw_buffers[buffer.index()].unwrap_or_else(|| {
            panic!(
                "Missing handle for '{}', is the buffer dead?",
                self.graph.input.get_buffer_display(buffer)
            )
        })
    }
    pub unsafe fn allocate_uniform_iter<T, I: IntoIterator<Item = T>>(
        &self,
        iter: I,
    ) -> UniformResult
    where
        I::IntoIter: ExactSizeIterator,
    {
        // this is RefMut, borrow checker cannot track struck fields as being borrowed separatly
        let state = self.graph.state();
        let res = state
            .descriptor_allocator
            .borrow_mut()
            .allocate_uniform_iter(&state.device, iter);
        res
    }
    pub unsafe fn allocate_uniform_element<T>(&self, value: T) -> UniformResult {
        let state = self.graph.state();
        let res = state
            .descriptor_allocator
            .borrow_mut()
            .allocate_uniform_element(&state.device, value);
        res
    }
    pub unsafe fn allocate_uniform_raw(&self, layout: std::alloc::Layout) -> UniformMemory {
        let state = self.graph.state();
        let res = state
            .descriptor_allocator
            .borrow_mut()
            .allocate_uniform_raw(&state.device, layout);
        res
    }
    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }
    pub unsafe fn bind_descriptor_sets(
        &self,
        bind_point: vk::PipelineBindPoint,
        layout: &ObjRef<object::PipelineLayout>,
        sets: &[&FinishedSet],
    ) {
        let state = self.graph.state();
        bind_descriptor_sets(
            &state.device.device(),
            self.command_buffer,
            bind_point,
            layout,
            sets,
        );
    }
}

/// The part of state needed by CompiledGraph that holds vulkan objects and must have its destruction deferred until work finishes
pub(crate) struct CompiledGraphVulkanState {
    pub(crate) passes: RefCell<Vec<PassObjectState>>,
    pub(crate) physical_images: Vec<PhysicalImageData>,
    pub(crate) physical_buffers: Vec<PhysicalBufferData>,
    pub(crate) allocations: Vec<pumice_vma::Allocation>,
    // semaphores used for presenting:
    //  to synchronize the moment when we aren't using the swapchain image anymore
    //  to start the submission only after the presentation engine is done with it
    // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Rendering_and_presentation#page_Creating-the-synchronization-objects
    pub(crate) swapchain_semaphores: RefCell<LegacySemaphoreStack>,
    pub(crate) descriptor_allocator: RefCell<DescriptorAllocator>,
    pub(crate) command_pools: RefCell<CommandBufferStack>,
    pub(crate) device: OwnedDevice,
}

impl CompiledGraphVulkanState {
    fn new(
        device: OwnedDevice,
        suballocators: &[Suballocator],
        passes: Vec<PassObjectState>,
        physical_images: Vec<PhysicalImageData>,
        physical_buffers: Vec<PhysicalBufferData>,
    ) -> Self {
        let allocations = suballocators
            .into_iter()
            .flat_map(|m| m.collect_blocks())
            .map(|b| b.0.allocation)
            .collect::<Vec<_>>();

        Self {
            swapchain_semaphores: RefCell::new(LegacySemaphoreStack::new()),
            command_pools: RefCell::new(CommandBufferStack::new()),
            allocations,
            passes: RefCell::new(passes),
            physical_buffers,
            physical_images,
            device,
            descriptor_allocator: RefCell::new(DescriptorAllocator::new()),
        }
    }
    unsafe fn reset(&mut self) {
        let device = &self.device;
        self.swapchain_semaphores.get_mut().reset();
        self.descriptor_allocator.get_mut().reset(device);
        self.command_pools.get_mut().reset(device);
    }
    unsafe fn destroy(&mut self) {
        let Self {
            // the pases are here only to drop their handles when they're not in flight anymore
            // TODO consider making an api where passes have to return their handles on destruction
            passes,
            physical_images,
            physical_buffers,
            allocations,
            descriptor_allocator,
            swapchain_semaphores,
            command_pools,
            device,
        } = self;

        let d = device.device();
        let ac = device.allocator_callbacks();

        descriptor_allocator.get_mut().destroy(device);
        swapchain_semaphores.get_mut().destroy(device);
        command_pools.get_mut().destroy(device);

        for i in physical_images {
            i.state.borrow_mut().destroy(device);
            d.destroy_image(i.vkhandle, ac);
        }

        for b in physical_buffers {
            b.state.borrow_mut().destroy(device);
            d.destroy_buffer(b.vkhandle, ac);
        }

        for a in allocations {
            device.allocator().free_memory(*a);
        }
    }
}

#[derive(Clone)]
struct SharedCompiledGraphVulkanState {
    state: ManuallyDrop<Arc<SendUnsafeCell<CompiledGraphVulkanState>>>,
}

impl Drop for SharedCompiledGraphVulkanState {
    fn drop(&mut self) {
        let arc = unsafe { ManuallyDrop::take(&mut self.state) };
        if let Ok(mut state) = Arc::try_unwrap(arc) {
            unsafe { state.get_mut().destroy() };
        }
    }
}

pub(crate) struct MainCompiledGraphVulkanState {
    shared: RefCell<SharedCompiledGraphVulkanState>,
}

impl MainCompiledGraphVulkanState {
    pub(crate) fn new(
        device: OwnedDevice,
        suballocators: &[Suballocator],
        passes: Vec<PassObjectState>,
        physical_images: Vec<PhysicalImageData>,
        physical_buffers: Vec<PhysicalBufferData>,
    ) -> Self {
        Self {
            shared: RefCell::new(SharedCompiledGraphVulkanState {
                state: ManuallyDrop::new(Arc::new(SendUnsafeCell::new(
                    CompiledGraphVulkanState::new(
                        device,
                        suballocators,
                        passes,
                        physical_images,
                        physical_buffers,
                    ),
                ))),
            }),
        }
    }
    #[track_caller]
    fn get(&self) -> Ref<CompiledGraphVulkanState> {
        Ref::map(self.shared.borrow(), |shared| unsafe { shared.state.get() })
    }
    #[track_caller]
    fn get_mut(&self) -> RefMut<CompiledGraphVulkanState> {
        RefMut::map(self.shared.borrow_mut(), |shared| unsafe {
            shared.state.get_mut()
        })
    }
    fn device(&self) -> &Device {
        // Device is internally synchronized, all its methods use an immutable reference
        unsafe { &(*self.shared.as_ptr()).state.get().device }
    }
    fn make_shared(&self) -> SharedCompiledGraphVulkanState {
        self.shared.borrow().clone()
    }
    fn make_finalizer(&self) -> Box<dyn FnOnce() + Send> {
        let shared = self.make_shared();
        Box::new(
            // this immediatelly closure drops shared when called
            move || {
                // interesting fact: if we use a `let _ = shared;` statement, the closure doesn't capture the variable?
                let friend = shared;
            },
        )
    }
}

pub struct CompiledGraph {
    pub(crate) input: CompilationInput,

    pub(crate) timeline: Vec<GraphPassEvent>,
    pub(crate) submissions: Vec<Submission>,

    pub(crate) external_resource_initial_access:
        HashMap<CombinedResourceHandle, ResourceFirstAccess>,
    pub(crate) image_last_state: Vec<ResourceState<ImageMarker>>,
    pub(crate) buffer_last_state: Vec<ResourceState<BufferMarker>>,

    pub(crate) alive_passes: FixedBitSet,
    pub(crate) alive_images: FixedBitSet,
    pub(crate) alive_buffers: FixedBitSet,

    pub(crate) current_generation: Cell<Option<GenerationId>>,
    pub(crate) prev_generation: Cell<Option<GenerationId>>,

    pub(crate) state: MainCompiledGraphVulkanState,
}

unsafe impl Send for CompiledGraphVulkanState {}

impl CompiledGraph {
    pub(crate) fn is_pass_alive(&self, pass: GraphPass) -> bool {
        self.alive_passes.contains(pass.index())
    }
    pub(crate) fn is_image_alive(&self, image: GraphImage) -> bool {
        self.alive_images.contains(image.index())
    }
    pub(crate) fn is_buffer_alive(&self, buffer: GraphBuffer) -> bool {
        self.alive_buffers.contains(buffer.index())
    }
    // FIXME code duplication with compile.rs
    pub(crate) fn get_image_create_info(&self, image: GraphImage) -> ImageKindCreateInfo<'_> {
        let mut data = self.input.get_image_data(image);
        loop {
            match data {
                ImageData::TransientPrototype(info, _) => {
                    return ImageKindCreateInfo::Image(info);
                }
                ImageData::Transient(physical) => {
                    return ImageKindCreateInfo::ImageRef(Ref::map(
                        self.get_physical_image(*physical),
                        |phys| &phys.info,
                    ));
                }
                ImageData::Imported(archandle) => {
                    return ImageKindCreateInfo::Image(unsafe { archandle.0.get_create_info() });
                }
                ImageData::Swapchain(archandle) => {
                    return ImageKindCreateInfo::Swapchain(unsafe {
                        archandle.0.get_create_info()
                    });
                }
                ImageData::Moved(_) => {
                    data = self.input.get_concrete_image_data(image);
                    continue;
                }
            }
        }
    }
    pub(crate) fn get_physical_image(&self, image: PhysicalImage) -> Ref<'_, PhysicalImageData> {
        Ref::map(self.state(), |state| &state.physical_images[image.index()])
    }
    pub(crate) fn get_physical_buffer(
        &self,
        buffer: PhysicalBuffer,
    ) -> Ref<'_, PhysicalBufferData> {
        Ref::map(self.state(), |state| {
            &state.physical_buffers[buffer.index()]
        })
    }
    #[track_caller]
    pub(crate) fn state(&self) -> Ref<'_, CompiledGraphVulkanState> {
        self.state.get()
    }
    #[track_caller]
    pub(crate) fn state_mut(&self) -> RefMut<'_, CompiledGraphVulkanState> {
        self.state.get_mut()
    }
    pub(crate) fn device(&self) -> &Device {
        self.state.device()
    }
    pub fn run(&mut self) {
        // we can now start executing the graph (TODO move this to another function, allow the built graph to be executed multiple times)
        // wait for previously submitted work to finish
        if let Some(id) = self.prev_generation.get() {
            self.state()
                .device
                .wait_for_generation_single(id, u64::MAX)
                .unwrap();
        }

        // since we've now waited for all previous work to finish, we can safely reset the semaphores
        unsafe {
            self.state_mut().reset();
        }

        // each submission gets a semaphore
        // TODO use sequential semaphore allocation to improve efficiency
        let submission_finish_semaphores = self
            .submissions
            .iter()
            .map(|s| {
                self.state()
                    .device
                    .make_submission(self.input.get_queue_family(s.queue), None)
            })
            .collect::<Vec<_>>();

        let state = self.state();

        // TODO when we separate compiling and executing a graph, these two loops will be far away
        for (i, pass) in self.state().passes.borrow_mut().iter_mut().enumerate() {
            if self.alive_passes.contains(i) {
                pass.on_created(|p| p.prepare());
            }
        }

        // TODO do something about this so that we don't hold the lock for all of execution
        let image_storage_lock = state.device.image_storage.acquire_all_exclusive();
        let buffer_storage_lock = state.device.buffer_storage.acquire_all_exclusive();
        let swapchain_storage_lock = state.device.swapchain_storage.acquire_all_exclusive();

        let mut waited_idle = false;
        let mut swapchain_image_indices: ahash::HashMap<GraphImage, u32> = constant_ahash_hashmap();
        let mut submission_swapchains: ahash::HashMap<QueueSubmission, Vec<SwapchainPresent>> =
            constant_ahash_hashmap();

        let mut accessors_scratch = Vec::new();

        let mut dummy_submissions = Vec::new();

        let mut submission_extra: ahash::HashMap<GraphSubmission, SubmissionExtra> =
            constant_ahash_hashmap();

        for (&handle, initial) in &self.external_resource_initial_access {
            let &ResourceFirstAccess {
                ref accessors,
                dst_layout,
                dst_queue_family,
                dst_stages,
                dst_access,
            } = initial;

            accessors_scratch.clear();
            accessors_scratch.extend(
                accessors
                    .iter()
                    .map(|a| submission_finish_semaphores[a.index()].0),
            );

            match handle.unpack() {
                GraphResource::Image(image) => {
                    let data = self.input.get_image_data(image);
                    match data {
                        ImageData::Imported(archandle) => unsafe {
                            let ResourceState::Normal(ResourceSubresource { layout, ..  }) = self.image_last_state[image.index()] else {
                                panic!("Unsupported state for external resource");
                            };

                            let concurrent = archandle.0.get_create_info().sharing_mode_concurrent;
                            let state = archandle.0.get_object_data().get_mutable_state();

                            self.handle_imported_sync(
                                handle,
                                &accessors_scratch,
                                initial,
                                concurrent,
                                layout,
                                state
                                    .get_mut(&image_storage_lock)
                                    .get_synchronization_state(),
                                &mut dummy_submissions,
                                &mut submission_extra,
                            );
                        },
                        ImageData::Swapchain(archandle) => unsafe {
                            assert!(accessors_scratch.len() == 1, "Swapchains use legacy semaphores and do not support multiple signals or waits, using a swapchain in multiple submissions is disallowed (you should really just transfer the final image into it at the end of the frame)");

                            let mut mutable = archandle
                                .0
                                .get_object_data()
                                .get_mutable_state(&swapchain_storage_lock);

                            let mut semaphores = state.swapchain_semaphores.borrow_mut();
                            let acquire_semaphore = semaphores.next(&state.device);
                            let release_semaphore = semaphores.next(&state.device);

                            let mut attempt = 0;
                            let index = loop {
                                match mutable.acquire_image(
                                    u64::MAX,
                                    acquire_semaphore,
                                    vk::Fence::null(),
                                    &self.state().device,
                                ) {
                                    SwapchainAcquireStatus::Ok(index, false) => break index,
                                    SwapchainAcquireStatus::OutOfDate
                                    | SwapchainAcquireStatus::Ok(_, true) => {
                                        // we allow the swapchain to be recreated two times, then crash
                                        if attempt == 2 {
                                            panic!("Swapchain immediatelly invalid after being recreated two times");
                                        }

                                        if !waited_idle {
                                            self.state().device.wait_idle();
                                            waited_idle = true;
                                        }

                                        mutable
                                            .recreate(
                                                &archandle.0.get_create_info(),
                                                &self.state().device,
                                            )
                                            .unwrap();

                                        attempt += 1;
                                        continue;
                                    }
                                    SwapchainAcquireStatus::Timeout => {
                                        panic!("Timeout when waiting u64::MAX")
                                    }
                                    SwapchainAcquireStatus::Err(e) => {
                                        panic!("Error acquiring swapchain image: {:?}", e)
                                    }
                                };
                            };

                            submission_swapchains
                                .entry(accessors_scratch[0])
                                .or_default()
                                .push(SwapchainPresent {
                                    vkhandle: mutable.get_swapchain(index),
                                    image_index: index,
                                    image_acquire: acquire_semaphore,
                                    image_release: release_semaphore,
                                });

                            // we do not update any synchronization state here, since we already get synchronization from swapchain image acquire and the subsequent present
                            swapchain_image_indices.insert(image, index);
                        },
                        _ => unreachable!("Not an external resource"),
                    }
                }
                GraphResource::Buffer(buffer) => {
                    let data = self.input.get_buffer_data(buffer);
                    match data {
                        BufferData::Imported(archandle) => unsafe {
                            let ResourceState::Normal(_) = self.buffer_last_state[buffer.index()] else {
                                panic!("Unsupported state for external resource");
                            };

                            let concurrent = archandle.0.get_create_info().sharing_mode_concurrent;
                            let state = archandle.0.get_object_data().get_mutable_state();

                            self.handle_imported_sync(
                                handle,
                                &accessors_scratch,
                                initial,
                                concurrent,
                                TypeNone::new_none(),
                                state
                                    .get_mut(&image_storage_lock)
                                    .get_synchronization_state(),
                                &mut dummy_submissions,
                                &mut submission_extra,
                            );
                        },
                        _ => {}
                    }
                }
            }
        }

        drop(image_storage_lock);
        drop(buffer_storage_lock);

        let mut deferrred_moved_images = Vec::new();

        let mut raw_images = self
            .input
            .images
            .iter()
            .enumerate()
            .map(|(i, data)| unsafe {
                let image = GraphImage::new(i);

                if !self.alive_images.contains(i) {
                    return None;
                }

                match data.deref() {
                    ImageData::TransientPrototype(_, _) => None,
                    ImageData::Transient(physical) => {
                        Some(self.get_physical_image(*physical).vkhandle)
                    }
                    ImageData::Imported(archandle) => Some(archandle.0.get_handle()),
                    ImageData::Swapchain(archandle) => {
                        let image_index = *swapchain_image_indices.get(&image).unwrap();
                        Some(
                            archandle
                                .0
                                .get_object_data()
                                .get_mutable_state(&swapchain_storage_lock)
                                .get_image_data(image_index)
                                .image,
                        )
                    }
                    ImageData::Moved(_) => match self.input.get_concrete_image_data(image) {
                        ImageData::Transient(physical) => {
                            Some(self.get_physical_image(*physical).vkhandle)
                        }
                        ImageData::Moved(_) | ImageData::TransientPrototype(_, _) => unreachable!(),
                        ImageData::Imported(_) | ImageData::Swapchain(_) => {
                            panic!("External images cannot be moved")
                        }
                    },
                }
            })
            .collect::<Vec<_>>();

        drop(swapchain_storage_lock);

        for moved in deferrred_moved_images {
            let move_dst = self.input.get_concrete_image_handle(moved);
            raw_images[moved.index()] = raw_images[move_dst.index()];
        }

        let raw_buffers = self
            .input
            .buffers
            .iter()
            .enumerate()
            .map(|(i, data)| unsafe {
                let buffer = GraphBuffer::new(i);

                if !self.alive_buffers.contains(i) {
                    return None;
                }

                match data.deref() {
                    BufferData::TransientPrototype(_, _) => None,
                    BufferData::Transient(physical) => {
                        Some(self.get_physical_buffer(*physical).vkhandle)
                    }
                    BufferData::Imported(archandle) => Some(archandle.0.get_handle()),
                }
            })
            .collect::<Vec<_>>();

        let id = {
            let gen = self
                .device()
                .open_generation(Some(self.state.make_finalizer()));
            gen.add_submissions(submission_finish_semaphores.iter().map(|(s, _)| *s));
            gen.finish()
        };

        self.current_generation.set(Some(id));

        // TODO multithreaded execution
        let mut submit_infos = Vec::new();

        let mut wait_semaphores = Vec::new();
        let mut signal_semaphores = Vec::new();

        let mut raw_memory_barriers = Vec::new();
        let mut raw_image_barriers = Vec::new();
        let mut raw_buffer_barriers = Vec::new();

        let bump = Bump::new();

        unsafe {
            let state = self.state();
            let mut command_pools = state.command_pools.borrow_mut();
            let d = self.device().device();

            dummy_submissions.sort_unstable_by_key(|s| s.queue);

            use slice_group_by::BinaryGroupByKey;
            for dummy_submission_group in dummy_submissions.binary_group_by_key(|s| s.queue) {
                submit_infos.clear();

                for dummy in dummy_submission_group {
                    let wait_semaphores =
                        bump.alloc_slice_fill_iter(dummy.dependencies.iter().map(|timeline| {
                            vk::SemaphoreSubmitInfoKHR {
                                semaphore: timeline.raw,
                                value: timeline.value,
                                stage_mask: vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                                ..Default::default()
                            }
                        }));
                    let signal_semaphore = bump.alloc(vk::SemaphoreSubmitInfoKHR {
                        semaphore: dummy.finished_semaphore.1.raw,
                        value: dummy.finished_semaphore.1.value,
                        stage_mask: vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                        ..Default::default()
                    });

                    let mut command_buffer_info = std::ptr::null();

                    let contains_barriers = dummy.contains_barriers();
                    if contains_barriers {
                        let command_buffer = command_pools.next(dummy.queue_family, self.device());

                        command_buffer_info = bump.alloc(vk::CommandBufferSubmitInfoKHR {
                            command_buffer,
                            ..Default::default()
                        }) as *const _;

                        d.begin_command_buffer(
                            command_buffer,
                            &vk::CommandBufferBeginInfo {
                                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                                ..Default::default()
                            },
                        )
                        .unwrap();

                        self.do_barriers_whole(
                            d,
                            std::iter::empty(),
                            &dummy.image_barriers,
                            &dummy.buffer_barriers,
                            &mut raw_memory_barriers,
                            &mut raw_image_barriers,
                            &mut raw_buffer_barriers,
                            &raw_images,
                            &raw_buffers,
                            command_buffer,
                        );

                        d.end_command_buffer(command_buffer);
                    }

                    let submit = vk::SubmitInfo2KHR {
                        flags: vk::SubmitFlagsKHR::empty(),
                        wait_semaphore_info_count: wait_semaphores.len() as u32,
                        p_wait_semaphore_infos: wait_semaphores.as_ffi_ptr(),
                        command_buffer_info_count: contains_barriers.then_some(1).unwrap_or(0),
                        p_command_buffer_infos: command_buffer_info,
                        signal_semaphore_info_count: 1,
                        p_signal_semaphore_infos: signal_semaphore,
                        ..Default::default()
                    };

                    submit_infos.push(submit);
                }

                d.queue_submit_2_khr(
                    dummy_submission_group[0].queue,
                    &submit_infos,
                    vk::Fence::null(),
                )
                .map_err(|e| panic!("Submit err {:?}", e))
                .unwrap();
            }

            for (i, sub) in self.submissions.iter().enumerate() {
                let submission = GraphSubmission::new(i);
                let command_buffer =
                    command_pools.next(self.input.get_queue_family(sub.queue), self.device());

                let Submission {
                    queue: _,
                    passes,
                    semaphore_dependencies,
                    memory_barriers,
                    image_barriers,
                    buffer_barriers,
                } = sub;
                let SubmissionExtra {
                    image_barriers: extra_image_barriers,
                    buffer_barriers: extra_buffer_barriers,
                    dependencies: extra_dependencies,
                } = submission_extra.remove(&submission).unwrap_or_default();

                d.begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap();

                let mut memory_barriers = memory_barriers.iter().cloned();
                let mut image_barriers = extra_image_barriers
                    .into_iter()
                    .map(|info| (SubmissionPass(0), info))
                    .chain(image_barriers.iter().cloned());
                let mut buffer_barriers = extra_buffer_barriers
                    .into_iter()
                    .map(|info| (SubmissionPass(0), info))
                    .chain(buffer_barriers.iter().cloned());

                for (i, pass) in passes.into_iter().enumerate() {
                    self.do_barriers(
                        i,
                        d,
                        &mut memory_barriers,
                        &mut image_barriers,
                        &mut buffer_barriers,
                        &mut raw_memory_barriers,
                        &mut raw_image_barriers,
                        &mut raw_buffer_barriers,
                        &raw_images,
                        &raw_buffers,
                        command_buffer,
                    );

                    let executor = GraphExecutor {
                        graph: self,
                        command_buffer,
                        swapchain_image_indices: &swapchain_image_indices,
                        raw_images: &raw_images,
                        raw_buffers: &raw_buffers,
                    };
                    state.passes.borrow_mut()[pass.index()]
                        .on_created(|p| p.execute(&executor, &state.device))
                        .unwrap();
                }
                self.do_barriers(
                    usize::MAX,
                    d,
                    &mut memory_barriers,
                    &mut image_barriers,
                    &mut buffer_barriers,
                    &mut raw_memory_barriers,
                    &mut raw_image_barriers,
                    &mut raw_buffer_barriers,
                    &raw_images,
                    &raw_buffers,
                    command_buffer,
                );

                d.end_command_buffer(command_buffer).unwrap();

                let pass_data = &self.input.passes[i];
                let queue = self.input.queues[pass_data.queue.index()].inner.clone();
                let (queue_submission, finished_semaphore) = submission_finish_semaphores[i];
                let swapchains = submission_swapchains.get(&queue_submission);

                wait_semaphores.clear();
                signal_semaphores.clear();
                if let Some(swapchains) = swapchains {
                    wait_semaphores = swapchains
                        .iter()
                        .map(|s| vk::SemaphoreSubmitInfoKHR {
                            semaphore: s.image_acquire,
                            // not a timeline semaphore, value ignored
                            value: 0,
                            // TODO we should probably bother to track this information
                            // though then transitive reduction and such become invalid and have to be patched up ... eh
                            stage_mask: vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                            ..Default::default()
                        })
                        .collect::<Vec<_>>();
                    signal_semaphores = swapchains
                        .iter()
                        .map(|s| vk::SemaphoreSubmitInfoKHR {
                            semaphore: s.image_release,
                            // not a timeline semaphore, value ignored
                            value: 0,
                            stage_mask: vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                            ..Default::default()
                        })
                        .collect::<Vec<_>>();
                }
                wait_semaphores.extend(extra_dependencies.iter().map(|timeline| {
                    vk::SemaphoreSubmitInfoKHR {
                        semaphore: timeline.raw,
                        value: timeline.value,
                        stage_mask: vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                        ..Default::default()
                    }
                }));
                signal_semaphores.push(vk::SemaphoreSubmitInfoKHR {
                    semaphore: finished_semaphore.raw,
                    value: finished_semaphore.value,
                    stage_mask: vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                    ..Default::default()
                });
                for dep in semaphore_dependencies {
                    let semaphore = submission_finish_semaphores[dep.index()].1;
                    wait_semaphores.push(vk::SemaphoreSubmitInfoKHR {
                        semaphore: semaphore.raw,
                        value: semaphore.value,
                        stage_mask: vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                        ..Default::default()
                    });
                }

                let command_buffer_info = vk::CommandBufferSubmitInfoKHR {
                    command_buffer,
                    ..Default::default()
                };

                let submit = vk::SubmitInfo2KHR {
                    flags: vk::SubmitFlagsKHR::empty(),
                    wait_semaphore_info_count: wait_semaphores.len() as u32,
                    p_wait_semaphore_infos: wait_semaphores.as_ptr(),
                    command_buffer_info_count: 1,
                    p_command_buffer_infos: &command_buffer_info,
                    signal_semaphore_info_count: signal_semaphores.len() as u32,
                    p_signal_semaphore_infos: signal_semaphores.as_ptr(),
                    ..Default::default()
                };

                // TODO some either timeout or constant quota based submit batching
                d.queue_submit_2_khr(
                    queue.raw(),
                    std::slice::from_ref(&submit),
                    vk::Fence::null(),
                )
                .map_err(|e| panic!("Submit err {:?}", e))
                .unwrap();

                if let Some(swapchains) = swapchains {
                    let wait_semaphores = swapchains
                        .iter()
                        .map(|s| s.image_release)
                        .collect::<Vec<_>>();
                    let image_indices =
                        swapchains.iter().map(|s| s.image_index).collect::<Vec<_>>();
                    let swapchains = swapchains.iter().map(|s| s.vkhandle).collect::<Vec<_>>();

                    let mut results = vec![std::mem::zeroed(); swapchains.len()];

                    let present_info = vk::PresentInfoKHR {
                        wait_semaphore_count: wait_semaphores.len() as u32,
                        p_wait_semaphores: wait_semaphores.as_ptr(),
                        swapchain_count: swapchains.len() as u32,
                        p_swapchains: swapchains.as_ptr(),
                        p_image_indices: image_indices.as_ptr(),
                        p_results: results.as_mut_ptr(),
                        ..Default::default()
                    };

                    d.queue_present_khr(queue.raw(), &present_info).unwrap();
                    let result = Ok(vk::Result::SUCCESS);

                    for result in std::iter::once(result).chain(
                        results
                            .iter()
                            .map(|res| pumice::new_result(vk::Result::SUCCESS, *res)),
                    ) {
                        match result {
                            Ok(vk::Result::SUCCESS) => {}
                            // the window became suboptimal while we were rendering, nothing to be done, the swapchain will be recreated in the next loop
                            Ok(vk::Result::SUBOPTIMAL_KHR)
                            | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {}
                            Err(
                                err @ (vk::Result::ERROR_OUT_OF_HOST_MEMORY
                                | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                                | vk::Result::ERROR_DEVICE_LOST
                                | vk::Result::ERROR_SURFACE_LOST_KHR),
                            ) => panic!("Fatal error '{:?}' from queue_present_khr", err),
                            Ok(huh) | Err(huh) => {
                                panic!("queue_present_khr shouldn't return a result of '{:?}'", huh)
                            }
                        }
                    }
                }
            }
        }

        self.prev_generation.set(Some(id));
    }
    unsafe fn handle_imported_sync<T: ResourceMarker>(
        &self,
        handle: CombinedResourceHandle,
        last_access_submissions: &[QueueSubmission],
        first_access: &ResourceFirstAccess,
        sharing_mode_concurrent: bool,
        final_layout: T::IfImage<vk::ImageLayout>,
        sync: &mut SynchronizationState<T>,
        dummy_submissions: &mut Vec<DummySubmission>,
        submission_extra: &mut ahash::HashMap<GraphSubmission, SubmissionExtra>,
    ) where
        T::IfImage<vk::ImageLayout>: Eq + Copy,
    {
        let ResourceFirstAccess {
            ref accessors,
            dst_layout,
            dst_stages,
            dst_access,
            dst_queue_family,
        } = *first_access;

        let synchronize_result = sync.update_state(
            dst_queue_family,
            T::IfImage::new_some(dst_layout),
            last_access_submissions,
            final_layout,
            dst_queue_family,
            sharing_mode_concurrent,
        );

        let SynchronizeResult {
            transition_layout_from,
            transition_ownership_from,
            prev_access,
        } = synchronize_result;

        let mut active: SmallVec<[TimelineSemaphore; 8]> = SmallVec::new();

        // we need to synchronize against all prev_acess passes
        // this is essentially the same as in emit_family_ownership_transition
        if !prev_access.is_empty() {
            self.state().device.collect_active_submission_datas_map(
                prev_access.iter().cloned(),
                &mut active,
                |s| s.semaphore,
            );
        }

        if transition_layout_from.is_none()
            && transition_ownership_from.is_none()
            && active.is_empty()
        {
            return;
        }

        assert!(!accessors.is_empty());

        fn push_barrier(
            resource: CombinedResourceHandle,
            src_stages: vk::PipelineStageFlags2KHR,
            dst_stages: vk::PipelineStageFlags2KHR,
            src_access: vk::AccessFlags2KHR,
            dst_access: vk::AccessFlags2KHR,
            old_layout: vk::ImageLayout,
            new_layout: vk::ImageLayout,
            src_queue_family_index: u32,
            dst_queue_family_index: u32,
            image_barriers: &mut Vec<ImageBarrier>,
            buffer_barriers: &mut Vec<BufferBarrier>,
        ) {
            match resource.unpack() {
                GraphResource::Image(image) => {
                    image_barriers.push(ImageBarrier {
                        image,
                        src_stages,
                        dst_stages,
                        src_access,
                        dst_access,
                        old_layout,
                        new_layout,
                        src_queue_family_index,
                        dst_queue_family_index,
                    });
                }
                GraphResource::Buffer(buffer) => {
                    buffer_barriers.push(BufferBarrier {
                        buffer,
                        src_stages,
                        dst_stages,
                        src_access,
                        dst_access,
                        src_queue_family_index,
                        dst_queue_family_index,
                    });
                }
            }
        }

        if transition_layout_from.is_some() || transition_ownership_from.is_some() {
            // create a dummy submission which will transition the ownership
            if let Some(src_queue_family) = transition_ownership_from {
                let finished_semaphore =
                    self.state().device.make_submission(src_queue_family, None);

                let mut release = DummySubmission {
                    queue_family: src_queue_family,
                    queue: self
                        .state()
                        .device
                        .find_queue_for_family(src_queue_family)
                        .unwrap(),
                    image_barriers: Vec::new(),
                    buffer_barriers: Vec::new(),
                    dependencies: SmallVec::from_slice(&active),
                    finished_semaphore,
                };
                push_barrier(
                    handle,
                    vk::PipelineStageFlags2KHR::empty(),
                    vk::PipelineStageFlags2KHR::empty(),
                    vk::AccessFlags2KHR::empty(),
                    vk::AccessFlags2KHR::empty(),
                    transition_layout_from.unwrap_or(dst_layout),
                    dst_layout,
                    src_queue_family,
                    dst_queue_family,
                    &mut release.image_barriers,
                    &mut release.buffer_barriers,
                );
                dummy_submissions.push(release);

                active.clear();
                active.push(finished_semaphore.1);
            }

            let extra = submission_extra.entry(accessors[0]).or_default();

            // we can just plop the acquire barrier onto the accessor submission
            if accessors.len() == 1 {
                push_barrier(
                    handle,
                    // it would possibly be beneficial to track the actual stages and access here?
                    // since src_stages is empty, does this actually define any unwanted execution dependency?
                    vk::PipelineStageFlags2KHR::empty(),
                    vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                    vk::AccessFlags2KHR::empty(),
                    vk::AccessFlags2KHR::all(),
                    transition_layout_from.unwrap_or(dst_layout),
                    dst_layout,
                    transition_ownership_from.unwrap_or(dst_queue_family),
                    dst_queue_family,
                    &mut extra.image_barriers,
                    &mut extra.buffer_barriers,
                );
            }
            // we need to create another dummy submission which will do the acquire and then all consumers will depend on it
            else {
                let finished_semaphore =
                    self.state().device.make_submission(dst_queue_family, None);

                let mut release = DummySubmission {
                    queue_family: dst_queue_family,
                    queue: self
                        .state()
                        .device
                        .find_queue_for_family(dst_queue_family)
                        .unwrap(),
                    image_barriers: Vec::new(),
                    buffer_barriers: Vec::new(),
                    dependencies: SmallVec::from_slice(&active),
                    finished_semaphore,
                };
                push_barrier(
                    handle,
                    vk::PipelineStageFlags2KHR::empty(),
                    vk::PipelineStageFlags2KHR::empty(),
                    vk::AccessFlags2KHR::empty(),
                    vk::AccessFlags2KHR::empty(),
                    transition_layout_from.unwrap_or(dst_layout),
                    dst_layout,
                    transition_ownership_from.unwrap_or(dst_queue_family),
                    dst_queue_family,
                    &mut release.image_barriers,
                    &mut release.buffer_barriers,
                );
                dummy_submissions.push(release);

                active.clear();
                active.push(finished_semaphore.1);
            }
        }

        for &submission in accessors {
            let extra = submission_extra.entry(submission).or_default();
            extra.dependencies.extend(active.iter().copied());
        }
    }
    unsafe fn do_barriers(
        &self,
        i: usize,
        d: &DeviceWrapper,
        memory_barriers: &mut (impl Iterator<Item = (SubmissionPass, MemoryBarrier)> + Clone),
        image_barriers: &mut (impl Iterator<Item = (SubmissionPass, ImageBarrier)> + Clone),
        buffer_barriers: &mut (impl Iterator<Item = (SubmissionPass, BufferBarrier)> + Clone),

        raw_memory_barriers: &mut Vec<vk::MemoryBarrier2KHR>,
        raw_image_barriers: &mut Vec<vk::ImageMemoryBarrier2KHR>,
        raw_buffer_barriers: &mut Vec<vk::BufferMemoryBarrier2KHR>,

        raw_images: &[Option<vk::Image>],
        raw_buffers: &[Option<vk::Buffer>],
        command_buffer: vk::CommandBuffer,
    ) {
        fn map_collect_into<'a, T: 'a, A, F: Fn(&T) -> A>(
            i: usize,
            from: &mut (impl Iterator<Item = (SubmissionPass, T)> + Clone),
            into: &mut Vec<A>,
            map: F,
        ) {
            into.clear();
            while let Some((pass, next)) = from.clone().next() {
                if pass.index() <= i {
                    from.next();
                    let new = map(&next);
                    into.push(new);
                } else {
                    break;
                }
            }
        }

        map_collect_into(
            i,
            memory_barriers,
            raw_memory_barriers,
            MemoryBarrier::to_vk,
        );

        map_collect_into(i, image_barriers, raw_image_barriers, |info| {
            let usage = match self.get_image_create_info(info.image) {
                ImageKindCreateInfo::ImageRef(i) => i.usage,
                ImageKindCreateInfo::Image(i) => i.usage,
                ImageKindCreateInfo::Swapchain(i) => i.usage,
            };

            let aspect = if usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            } else {
                vk::ImageAspectFlags::COLOR
            };

            info.to_vk(
                raw_images[info.image.index()].unwrap(),
                self.input.get_image_subresource_range(info.image, aspect),
            )
        });

        map_collect_into(i, buffer_barriers, raw_buffer_barriers, |info| {
            info.to_vk(raw_buffers[info.buffer.index()].unwrap())
        });

        if !(raw_memory_barriers.is_empty()
            && raw_image_barriers.is_empty()
            && raw_buffer_barriers.is_empty())
        {
            d.cmd_pipeline_barrier_2_khr(
                command_buffer,
                &vk::DependencyInfoKHR {
                    // TODO track opportunities to use BY_REGION and friends
                    // https://stackoverflow.com/questions/65471677/the-meaning-and-implications-of-vk-dependency-by-region-bit
                    dependency_flags: vk::DependencyFlags::empty(),
                    memory_barrier_count: raw_memory_barriers.len() as u32,
                    p_memory_barriers: raw_memory_barriers.as_ffi_ptr(),
                    buffer_memory_barrier_count: raw_buffer_barriers.len() as u32,
                    p_buffer_memory_barriers: raw_buffer_barriers.as_ffi_ptr(),
                    image_memory_barrier_count: raw_image_barriers.len() as u32,
                    p_image_memory_barriers: raw_image_barriers.as_ffi_ptr(),
                    ..Default::default()
                },
            );
        }
    }
    unsafe fn do_barriers_whole<'a>(
        &self,
        d: &DeviceWrapper,
        memory_barriers: impl IntoIterator<Item = &'a MemoryBarrier>,
        image_barriers: impl IntoIterator<Item = &'a ImageBarrier>,
        buffer_barriers: impl IntoIterator<Item = &'a BufferBarrier>,

        raw_memory_barriers: &mut Vec<vk::MemoryBarrier2KHR>,
        raw_image_barriers: &mut Vec<vk::ImageMemoryBarrier2KHR>,
        raw_buffer_barriers: &mut Vec<vk::BufferMemoryBarrier2KHR>,

        raw_images: &[Option<vk::Image>],
        raw_buffers: &[Option<vk::Buffer>],
        command_buffer: vk::CommandBuffer,
    ) {
        fn collect_into<'a, T: 'a, A, F: Fn(&T) -> A>(
            from: impl IntoIterator<Item = &'a T>,
            into: &mut Vec<A>,
            map: F,
        ) {
            into.clear();
            into.extend(from.into_iter().map(map));
        }

        collect_into(memory_barriers, raw_memory_barriers, MemoryBarrier::to_vk);

        collect_into(image_barriers, raw_image_barriers, |info| {
            info.to_vk(
                raw_images[info.image.index()].unwrap(),
                self.input
                    .get_image_subresource_range(info.image, vk::ImageAspectFlags::all()),
            )
        });

        collect_into(buffer_barriers, raw_buffer_barriers, |info| {
            info.to_vk(raw_buffers[info.buffer.index()].unwrap())
        });

        if !(raw_memory_barriers.is_empty()
            && raw_image_barriers.is_empty()
            && raw_buffer_barriers.is_empty())
        {
            d.cmd_pipeline_barrier_2_khr(
                command_buffer,
                &vk::DependencyInfoKHR {
                    dependency_flags: vk::DependencyFlags::empty(),
                    memory_barrier_count: raw_memory_barriers.len() as u32,
                    p_memory_barriers: raw_memory_barriers.as_ffi_ptr(),
                    buffer_memory_barrier_count: raw_buffer_barriers.len() as u32,
                    p_buffer_memory_barriers: raw_buffer_barriers.as_ffi_ptr(),
                    image_memory_barrier_count: raw_image_barriers.len() as u32,
                    p_image_memory_barriers: raw_image_barriers.as_ffi_ptr(),
                    ..Default::default()
                },
            );
        }
    }
}

struct DummySubmission {
    queue_family: u32,
    queue: vk::Queue,
    image_barriers: Vec<ImageBarrier>,
    buffer_barriers: Vec<BufferBarrier>,
    dependencies: SmallVec<[TimelineSemaphore; 1]>,
    finished_semaphore: (QueueSubmission, TimelineSemaphore),
}

impl DummySubmission {
    fn contains_barriers(&self) -> bool {
        !self.image_barriers.is_empty() || !self.buffer_barriers.is_empty()
    }
}
struct SubmissionExtra {
    image_barriers: Vec<ImageBarrier>,
    buffer_barriers: Vec<BufferBarrier>,
    dependencies: ahash::HashSet<TimelineSemaphore>,
}

impl Default for SubmissionExtra {
    fn default() -> Self {
        Self {
            image_barriers: Default::default(),
            buffer_barriers: Default::default(),
            dependencies: constant_ahash_hashset(),
        }
    }
}
