mod allocator;
mod lazy_sorted;
pub mod passes;
pub mod resource_marker;
mod reverse_edges;

use std::{
    borrow::{Borrow, Cow},
    cell::{Cell, RefCell, RefMut},
    collections::{hash_map::Entry, BinaryHeap, HashSet, VecDeque},
    f32::consts::E,
    fmt::Display,
    fs::OpenOptions,
    hash::{Hash, Hasher},
    io::Write,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{ControlFlow, Deref, DerefMut, Not, Range},
    sync::Arc,
};

use ahash::HashMap;
use pumice::{util::ObjectHandle, vk, vk10::CommandPoolCreateInfo, DeviceWrapper, VulkanResult};
use pumice_vma::{Allocation, AllocationCreateInfo};
use smallvec::{smallvec, SmallVec};

use crate::{
    arena::uint::{Config, OptionalU32, PackedUint},
    device::{
        Device, OwnedDevice, __test_init_device,
        batch::GenerationId,
        inflight::InflightResource,
        submission::{self, QueueSubmission},
    },
    graph::{
        allocator::MemoryKind,
        resource_marker::{BufferMarker, ImageMarker, TypeNone, TypeOption, TypeSome},
        reverse_edges::{reverse_edges_into, ChildRelativeKey, NodeKey},
    },
    object::{
        self, BufferCreateInfo, ImageCreateInfo, Object, ResourceMutableState,
        SwapchainAcquireStatus,
    },
    storage::{
        constant_ahash_hasher, constant_ahash_hashmap, constant_ahash_hashset,
        constant_ahash_randomstate, ObjectStorage, SynchronizationLock,
    },
    token_abuse,
    util::{self, format_utils::Fun, macro_abuse::WeirdFormatter},
};

use self::{
    allocator::{
        AvailabilityToken, RcPtrComparator, Suballocation, SuballocationUgh, Suballocator,
    },
    lazy_sorted::LazySorted,
    resource_marker::{ResourceData, ResourceMarker},
    reverse_edges::{DFSCommand, ImmutableGraph, NodeGraph},
};

pub trait RenderPass: 'static {
    fn prepare(&mut self);
    unsafe fn execute(self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()>;
}

impl RenderPass for () {
    fn prepare(&mut self) {
        {}
    }
    unsafe fn execute(self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()> {
        VulkanResult::Ok(())
    }
}

pub trait CreatePass {
    type Pass: RenderPass;
    fn create(self, builder: &mut GraphPassBuilder, device: &Device) -> Self::Pass;
}
impl<P: RenderPass, F: FnOnce(&mut GraphPassBuilder, &Device) -> P> CreatePass for F {
    type Pass = P;
    fn create(self, builder: &mut GraphPassBuilder, device: &Device) -> Self::Pass {
        self(builder, device)
    }
}

struct StoredPass<T: RenderPass>(Option<T>);
trait ObjectSafePass {
    fn prepare(&mut self);
    unsafe fn execute(&mut self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()>;
}
impl<T: RenderPass> ObjectSafePass for StoredPass<T> {
    fn prepare(&mut self) {
        self.0.as_mut().unwrap().prepare()
    }
    unsafe fn execute(&mut self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()> {
        self.0.take().unwrap().execute(executor, device)
    }
}

#[derive(Clone)]
struct PassMeta {
    alive: Cell<bool>,
    scheduled_submission: Cell<OptionalU32>,
    scheduled_submission_position: Cell<OptionalU32>,
    queue_intervals: Cell<QueueIntervals>,
}
#[derive(Clone)]
struct ImageMeta {
    alive: Cell<bool>,
    physical: OptionalU32,
    producer: Cell<GraphPassOption>,
}
impl ImageMeta {
    fn new() -> Self {
        Self {
            alive: Cell::new(false),
            physical: OptionalU32::NONE,
            producer: Cell::new(GraphPassOption::NONE),
        }
    }
}
#[derive(Clone)]
struct BufferMeta {
    alive: Cell<bool>,
    physical: OptionalU32,
    producer: Cell<GraphPassOption>,
}
impl BufferMeta {
    fn new() -> Self {
        Self {
            alive: Cell::new(false),
            physical: OptionalU32::NONE,
            producer: Cell::new(GraphPassOption::NONE),
        }
    }
}

#[derive(Clone)]
struct ImageBarrier {
    pub image: GraphImage,
    pub src_stage_mask: vk::PipelineStageFlags2KHR,
    pub src_access_mask: vk::AccessFlags2KHR,
    pub dst_stage_mask: vk::PipelineStageFlags2KHR,
    pub dst_access_mask: vk::AccessFlags2KHR,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
}

#[derive(Clone)]
struct BufferBarrier {
    pub buffer: vk::Buffer,
    pub src_access_mask: vk::AccessFlags,
    pub dst_access_mask: vk::AccessFlags,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
}

#[derive(Clone)]
struct MemoryBarrier {
    pub src_stage_mask: vk::PipelineStageFlags2KHR,
    pub src_access_mask: vk::AccessFlags2KHR,
    pub dst_stage_mask: vk::PipelineStageFlags2KHR,
    pub dst_access_mask: vk::AccessFlags2KHR,
}

#[derive(Clone)]
struct SimpleBarrier {
    pub src_stages: vk::PipelineStageFlags2KHR,
    pub src_access: vk::AccessFlags2KHR,
    pub dst_stages: vk::PipelineStageFlags2KHR,
    pub dst_access: vk::AccessFlags2KHR,
}

#[derive(Clone)]
enum SpecialBarrier {
    LayoutTransition {
        image: GraphImage,
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
    },
    ImageOwnershipTransition {
        image: GraphImage,
        src_family: u32,
        dst_family: u32,
    },
    BufferOwnershipTransition {
        buffer: GraphBuffer,
        src_family: u32,
        dst_family: u32,
    },
}

struct GraphMemoryType {
    allocator: Suballocator,
    memory_type: u32,
}

impl GraphMemoryType {
    fn new(memory_type: u32) -> Self {
        Self {
            allocator: Suballocator::new(),
            memory_type,
        }
    }
}

struct PhysicalImageData {
    info: ImageCreateInfo,
    memory: SuballocationUgh,
    vkhandle: vk::Image,
    state: ResourceMutableState<ImageMarker>,
}

impl PhysicalImageData {
    fn get_memory_type(&self) -> u32 {
        self.memory.memory.memory_type
    }
}

struct PhysicalBufferData {
    info: BufferCreateInfo,
    memory: SuballocationUgh,
    vkhandle: vk::Buffer,
    state: ResourceMutableState<BufferMarker>,
}

#[derive(Default)]
struct LegacySemaphoreStack {
    next_index: usize,
    semaphores: Vec<vk::Semaphore>,
}

impl LegacySemaphoreStack {
    pub fn new() -> Self {
        Self::default()
    }
    unsafe fn reset(&mut self) {
        self.next_index = 0;
    }
    unsafe fn next(&mut self, ctx: &Device) -> vk::Semaphore {
        if self.next_index == self.semaphores.len() {
            let info = vk::SemaphoreCreateInfo::default();
            let semaphore = ctx.create_raw_semaphore().unwrap();
            self.semaphores.push(semaphore);
        }

        let semaphore = self.semaphores[self.next_index];
        self.next_index += 1;
        semaphore
    }
    unsafe fn destroy(self, ctx: &Device) {
        for semaphore in self.semaphores {
            ctx.destroy_raw_semaphore(semaphore);
        }
    }
}

impl PhysicalBufferData {
    fn get_memory_type(&self) -> u32 {
        self.memory.memory.memory_type
    }
}

#[derive(Clone)]
struct Submission {
    queue: GraphQueue,
    passes: Vec<GraphPass>,
    semaphore_dependencies: Vec<GraphSubmission>,
    // barriers and special_barriers may have SubmissionPass targets that go beyond the actual passes vector
    // these are emitted to fixup synchronization for queue ownership transfer
    barriers: Vec<(SubmissionPass, SimpleBarrier)>,
    special_barriers: Vec<(SubmissionPass, SpecialBarrier)>,
}

pub struct Graph {
    queues: Vec<GraphObject<submission::Queue>>,
    images: Vec<GraphObject<ImageData>>,
    buffers: Vec<GraphObject<BufferData>>,

    timeline: Vec<GraphPassEvent>,
    passes: Vec<GraphObject<PassData>>,
    moves: Vec<ImageMove>,

    pass_meta: Vec<PassMeta>,
    image_meta: Vec<ImageMeta>,
    buffer_meta: Vec<BufferMeta>,

    pass_children: ImmutableGraph,

    // FIXME should this be made prettier?
    memory: RefCell<ManuallyDrop<Box<[GraphMemoryType; vk::MAX_MEMORY_TYPES as usize]>>>,
    physical_images: Vec<PhysicalImageData>,
    physical_buffers: Vec<PhysicalBufferData>,

    prev_generation: Cell<Option<GenerationId>>,
    // semaphores used for presenting:
    //  to synchronize the moment when we aren't using the swapchain image anymore
    //  to start the submission only after the presentation engine is done with it
    // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Rendering_and_presentation#page_Creating-the-synchronization-objects
    swapchain_semaphores: LegacySemaphoreStack,
    command_pool: vk::CommandPool,
    device: ManuallyDrop<OwnedDevice>,
}

impl Graph {
    pub fn new(device: OwnedDevice) -> Self {
        let memory = (0..vk::MAX_MEMORY_TYPES)
            .map(|i| GraphMemoryType::new(i))
            .collect::<Box<[_]>>();

        Graph {
            queues: Vec::new(),
            images: Vec::new(),
            buffers: Vec::new(),
            timeline: Vec::new(),
            passes: Vec::new(),
            moves: Vec::new(),
            pass_meta: Vec::new(),
            image_meta: Vec::new(),
            buffer_meta: Vec::new(),
            pass_children: Default::default(),
            memory: RefCell::new(ManuallyDrop::new(memory.try_into().ok().unwrap())),
            physical_images: Vec::new(),
            physical_buffers: Vec::new(),
            prev_generation: Cell::new(None),
            swapchain_semaphores: LegacySemaphoreStack::new(),
            command_pool: unsafe {
                device
                    .device()
                    .create_command_pool(
                        &CommandPoolCreateInfo::default(),
                        device.allocator_callbacks(),
                    )
                    .unwrap()
            },
            device: ManuallyDrop::new(device),
        }
    }
    fn mark_pass_alive(&self, handle: GraphPass) {
        let i = handle.index();
        let pass = &self.passes[i];
        let meta = &self.pass_meta[i];

        // if true, we have already touched its dependencies and can safely return
        if meta.alive.get() {
            return;
        }
        meta.alive.set(true);

        for i in &pass.images {
            self.mark_image_alive(i.handle);
        }
        for b in &pass.buffers {
            self.mark_buffer_alive(b.handle);
        }

        for p in &pass.dependencies {
            // only hard dependencies propagate aliveness
            if p.is_hard() {
                self.mark_pass_alive(p.get_pass());
            }
        }
    }
    fn mark_image_alive(&self, image: GraphImage) {
        let mut image = image;
        loop {
            let i = image.index();
            self.image_meta[i].alive.set(true);
            match &*self.images[i] {
                ImageData::Moved(to, ..) => {
                    image = *to;
                }
                _ => break,
            }
        }
    }
    fn mark_buffer_alive(&self, handle: GraphBuffer) {
        let i = handle.index();
        self.buffer_meta[i].alive.set(true);
    }
    fn clear(&mut self) {
        self.queues.clear();
        self.passes.clear();
        self.images.clear();
        self.buffers.clear();
    }
    fn prepare_meta(&mut self) {
        self.pass_meta.clear();
        self.image_meta.clear();
        self.buffer_meta.clear();

        let len = self.passes.len();

        self.pass_meta.resize(
            len,
            PassMeta {
                alive: Cell::new(false),
                scheduled_submission_position: Cell::new(OptionalU32::NONE),
                scheduled_submission: Cell::new(OptionalU32::NONE),
                queue_intervals: Cell::new(QueueIntervals::NONE),
            },
        );
        self.image_meta.resize(len, ImageMeta::new());
        self.buffer_meta.resize(len, BufferMeta::new());
    }
    fn get_concrete_image_data(&self, image: GraphImage) -> &ImageData {
        let mut image = image;
        loop {
            match self.get_image_data(image) {
                ImageData::Moved(to, ..) => {
                    image = *to;
                }
                other => return other,
            }
        }
    }
    fn is_image_external<'a>(&'a self, image: GraphImage) -> bool {
        match self.get_concrete_image_data(image) {
            ImageData::TransientPrototype(..) => false,
            ImageData::Imported(_) => true,
            ImageData::Swapchain(_) => true,
            ImageData::Transient(..) => unreachable!(),
            ImageData::Moved(..) => unreachable!(),
        }
    }
    fn is_buffer_external(&self, mut buffer: GraphBuffer) -> bool {
        match self.get_buffer_data(buffer) {
            BufferData::TransientPrototype(..) => false,
            BufferData::Transient(..) => unreachable!(),
            BufferData::Imported(_) => true,
        }
    }
    fn is_pass_alive(&self, pass: GraphPass) -> bool {
        self.pass_meta[pass.index()].alive.get()
    }
    fn get_suballocator(&self, memory_type: u32) -> RefMut<Suballocator> {
        RefMut::map(self.memory.borrow_mut(), |m| {
            &mut m[memory_type as usize].allocator
        })
    }
    fn get_image_data(&self, image: GraphImage) -> &ImageData {
        &self.images[image.index()]
    }
    fn get_physical_image_data(&self, image: PhysicalImage) -> &PhysicalImageData {
        &self.physical_images[image.index()]
    }
    fn get_image_meta(&self, image: GraphImage) -> &ImageMeta {
        &self.image_meta[image.index()]
    }
    fn get_image_data_mut(&mut self, image: GraphImage) -> &mut ImageData {
        &mut self.images[image.index()]
    }
    fn get_buffer_data(&self, buffer: GraphBuffer) -> &BufferData {
        &self.buffers[buffer.index()]
    }
    fn get_buffer_data_mut(&mut self, buffer: GraphBuffer) -> &mut BufferData {
        &mut self.buffers[buffer.index()]
    }
    fn get_physical_buffer_data(&self, buffer: PhysicalBuffer) -> &PhysicalBufferData {
        &self.physical_buffers[buffer.index()]
    }
    fn get_buffer_meta(&self, buffer: GraphBuffer) -> &BufferMeta {
        &self.buffer_meta[buffer.index()]
    }
    fn get_pass_data(&self, pass: GraphPass) -> &PassData {
        &self.passes[pass.0 as usize]
    }
    fn get_pass_meta(&self, pass: GraphPass) -> &PassMeta {
        &self.pass_meta[pass.0 as usize]
    }
    fn get_pass_move(&self, move_handle: GraphPassMove) -> &ImageMove {
        &self.moves[move_handle.index()]
    }
    fn get_start_passes<'a>(&'a self) -> impl Iterator<Item = GraphPass> + 'a {
        self.passes
            .iter()
            .enumerate()
            .filter(|&(i, pass)| self.pass_meta[i].alive.get() && pass.dependencies.is_empty())
            .map(|(i, _)| GraphPass::new(i))
    }
    fn get_alive_passes<'a>(&'a self) -> impl Iterator<Item = GraphPass> + 'a {
        self.pass_meta
            .iter()
            .enumerate()
            .filter(|&(i, meta)| meta.alive.get())
            .map(|(i, _)| GraphPass::new(i))
    }
    fn get_children(&self, pass: GraphPass) -> &[GraphPass] {
        let meta = &self.pass_meta[pass.index()];
        let children = self.pass_children.get_children(pass.0);
        // sound because handles are repr(transparent)
        unsafe {
            std::slice::from_raw_parts::<'_, GraphPass>(
                children.as_ptr() as *const GraphPass,
                children.len(),
            )
        }
    }
    fn get_dependencies(&self, pass: GraphPass) -> &[PassDependency] {
        &self.passes[pass.index()].dependencies
    }
    fn get_queue_family(&self, queue: GraphQueue) -> u32 {
        self.queues[queue.index()].family()
    }
    fn get_queue_families(&self) -> SmallVec<[u32; 4]> {
        let mut vec: SmallVec<[u32; 4]> = SmallVec::new();
        for queue in &self.queues {
            vec.push(queue.family());
        }
        vec.sort_unstable();
        vec.dedup();
        vec
    }
    fn get_queue_display(&self, queue: GraphQueue) -> GraphObjectDisplay<'_> {
        self.queues[queue.index()].display(queue.index())
    }
    fn get_pass_display(&self, pass: GraphPass) -> GraphObjectDisplay<'_> {
        self.passes[pass.index()].display(pass.index())
    }
    fn get_image_display(&self, image: GraphImage) -> GraphObjectDisplay<'_> {
        self.images[image.index()].display(image.index())
    }
    fn get_buffer_display(&self, buffer: GraphBuffer) -> GraphObjectDisplay<'_> {
        self.buffers[buffer.index()].display(buffer.index())
    }
    fn compute_graph_layer(&self, pass: GraphPass, graph_layers: &mut [i32]) -> i32 {
        // either it's -1 and is dead or has already been touched and has a positive number
        let layer = graph_layers[pass.index()];
        if layer == -1 {
            return -1;
        }
        if layer > 0 {
            return layer;
        }
        let max = self
            .get_dependencies(pass)
            .iter()
            .map(|&d| self.compute_graph_layer(d.get_pass(), graph_layers))
            .max()
            .unwrap();
        let current = max + 1;
        graph_layers[pass.index()] = current;
        current
    }
    pub fn run<F: FnOnce(&mut GraphBuilder)>(&mut self, fun: F) {
        self.clear();

        // get the graph from the user
        // sound because GraphBuilder is repr(transparent)
        let builder = unsafe { std::mem::transmute::<&mut Graph, &mut GraphBuilder>(self) };
        fun(builder);
        self.prepare_meta();

        // src dst
        //  W   W  -- hard
        //  W   R  -- hard
        //  R   W  -- soft
        //  R   R  -- nothing
        #[inline]
        const fn is_hard(src_writes: bool, dst_writes: bool) -> Option<bool> {
            match (src_writes, dst_writes) {
                (true, true) => Some(true),
                (true, false) => Some(true),
                (false, true) => Some(false),
                (false, false) => None,
            }
        }

        // replay timeline events and add dependency between passes that arise out of memory dependencies
        {
            #[derive(Default, Clone)]
            enum ResourceState {
                #[default]
                Uninit,
                MoveDst {
                    reading: SmallVec<[GraphPass; 8]>,
                    writing: SmallVec<[GraphPass; 8]>,
                },
                Normal {
                    reading: SmallVec<[GraphPass; 8]>,
                    writing: Option<GraphPass>,
                },
                Moved,
            }

            impl ResourceState {
                fn new_normal(accessor: GraphPass, writing: bool) -> Self {
                    if writing {
                        ResourceState::Normal {
                            reading: SmallVec::new(),
                            writing: Some(accessor),
                        }
                    } else {
                        ResourceState::Normal {
                            reading: smallvec![accessor],
                            writing: None,
                        }
                    }
                }
            }

            fn update_resource_state(
                src: &mut ResourceState,
                p: GraphPass,
                dst_writing: bool,
                data: &mut PassData,
            ) {
                match src {
                    // no dependency
                    ResourceState::Uninit => {
                        *src = ResourceState::new_normal(p, dst_writing);
                    }
                    // inherit all dependencies
                    ResourceState::MoveDst { reading, writing } => {
                        if let Some(is_hard) = is_hard(false, dst_writing) {
                            for r in reading {
                                data.add_dependency(*r, is_hard, false);
                            }
                        }
                        if let Some(is_hard) = is_hard(true, dst_writing) {
                            for r in writing {
                                data.add_dependency(*r, is_hard, false);
                            }
                        }
                    }
                    ResourceState::Normal { reading, writing } => {
                        // note that when a Write occurs and then some Reads, the next Write will
                        // make a hard dependency to the previous Write and soft dependencies to all the Reads

                        if let Some(producer) = writing {
                            assert!(reading.is_empty());
                            data.add_dependency(
                                *producer,
                                // src is WRITE, some dependency must occur
                                is_hard(true, dst_writing).unwrap(),
                                false,
                            );

                            // W W
                            if dst_writing {
                                *writing = Some(p);
                            }
                            // W R
                            else {
                                reading.push(p);
                            }
                        }
                        // R W
                        if dst_writing {
                            for r in &*reading {
                                data.add_dependency(
                                    *r,
                                    is_hard(false, /* dst_writing == */ true).unwrap(),
                                    false,
                                );
                            }
                            reading.clear();
                            *writing = Some(p);
                        }
                        // R R - we only append this pass to the current readers, no dependency is created
                        else {
                            if !reading.contains(&p) {
                                reading.push(p);
                            }
                        }
                    }
                    // TODO perhaps this shouldn't be a hard error and instead delegate access to the move destination
                    ResourceState::Moved => panic!("Attempt to access moved resource"),
                }
            }

            let mut image_rw = vec![ResourceState::default(); self.images.len()];
            let mut buffer_rw = vec![ResourceState::default(); self.buffers.len()];

            for &e in &self.timeline {
                match e.get() {
                    PassEventData::Pass(p) => {
                        let data = &mut self.passes[p.index()];

                        for i_index in 0..data.images.len() {
                            let image = &data.images[i_index];
                            let dst_writing = image.is_written();
                            let src = &mut image_rw[image.handle.index()];

                            update_resource_state(src, p, dst_writing, data);
                        }

                        for b_index in 0..data.buffers.len() {
                            let buffer = &data.buffers[b_index];
                            let dst_writing = buffer.is_written();
                            let src = &mut buffer_rw[buffer.handle.index()];

                            let resource = GraphResource::Buffer(buffer.handle);

                            update_resource_state(src, p, dst_writing, data);
                        }
                    }
                    PassEventData::Move(m) => {
                        let ImageMove { from, to } = self.get_pass_move(m);
                        // ResourceState will not have PartialEq, use this to compare varitants
                        let ResourceState::Uninit = image_rw[to.index()] else {
                            panic!("Move destinations is not Uninit!");
                        };

                        // TODO reuse allocations for this from the moved images
                        let mut reads = SmallVec::new();
                        let mut writes = SmallVec::new();

                        // collect dependencies from all constituent resources
                        for &i in from {
                            let ResourceState::Normal { reading, writing } = std::mem::replace(&mut image_rw[i.index()], ResourceState::Moved) else {
                                panic!("Resource in unsupported state");
                            };

                            for r in reading {
                                if !reads.contains(&r) {
                                    reads.push(r);
                                }
                            }
                            if let Some(r) = writing {
                                if let Some(found) = reads.iter().position(|&p| p == r) {
                                    reads.swap_remove(found);
                                }
                                if !writes.contains(&r) {
                                    writes.push(r);
                                }
                            }
                        }
                        image_rw[to.index()] = ResourceState::MoveDst {
                            reading: reads,
                            writing: writes,
                        };
                    }
                    // flushes constrain scheduling order but do not have any effect on resource state
                    PassEventData::Flush(_) => {}
                }
            }
        };

        // find any pass that writes to external resources, thus being considered to have side effects
        // outside of the graph and mark all of its dependencies as alive, any passes that don't get touched
        // are never scheduled and their resources never instantiated
        for (i, pass) in self.passes.iter().enumerate() {
            // if this pass is already alive, all of its dependendies must have been touched already and we have nothing to do
            if self.pass_meta[i].alive.get() {
                continue;
            }

            if pass.force_run
                || pass
                    .images
                    .iter()
                    .any(|i| i.is_written() && self.is_image_external(i.handle))
                || pass
                    .buffers
                    .iter()
                    .any(|p| p.is_written() && self.is_buffer_external(p.handle))
            {
                self.mark_pass_alive(GraphPass(i as u32));
            }
        }

        struct GraphFacade<'a>(&'a Graph, ());
        impl<'a> NodeGraph for GraphFacade<'a> {
            type NodeData = ();

            fn node_count(&self) -> usize {
                self.0.passes.len()
            }
            fn nodes(&self) -> Range<NodeKey> {
                0..self.0.passes.len() as u32
            }
            fn children(&self, this: NodeKey) -> Range<ChildRelativeKey> {
                0..self.0.passes[this as usize].dependencies.len() as u32
            }
            fn get_child(&self, this: NodeKey, child: ChildRelativeKey) -> NodeKey {
                self.0.passes[this as usize].dependencies[child as usize]
                    .get_pass()
                    .0
            }
            fn get_node_data(&self, this: NodeKey) -> &Self::NodeData {
                &()
            }
            fn get_node_data_mut(&mut self, this: NodeKey) -> &mut Self::NodeData {
                &mut self.1
            }
        }

        // collect the dependees of alive passes (edges going other way than dependencies)
        let mut graph = std::mem::take(&mut self.pass_children);
        let facade = GraphFacade(self, ());
        reverse_edges_into(&facade, &mut graph);
        std::mem::replace(&mut self.pass_children, graph);

        // do a greedy graph traversal where nodes with a larger priority will be selected first
        // at this point this is essentially just a bfs
        let scheduled = {
            #[derive(Clone)]
            struct AvailablePass {
                pass: GraphPass,
                priority: i32,
            }

            impl AvailablePass {
                fn new(pass: GraphPass, graph: &Graph, graph_layers: &mut [i32]) -> Self {
                    let layer = graph.compute_graph_layer(pass, graph_layers);
                    Self {
                        pass,
                        priority: -layer,
                    }
                }
            }

            impl PartialEq for AvailablePass {
                fn eq(&self, other: &Self) -> bool {
                    self.priority == other.priority
                }
            }
            impl PartialOrd for AvailablePass {
                fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                    self.priority.partial_cmp(&other.priority)
                }
            }
            impl Eq for AvailablePass {}
            impl Ord for AvailablePass {
                fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                    self.priority.cmp(&other.priority)
                }
            }

            // (u16, bool) <- flags whether the pass is in a currently active flush region
            //  ^ count of unscheduled dependencies
            let mut dependency_count = self
                .passes
                .iter()
                .map(|p| (p.dependencies.len() as u16, false))
                .collect::<Vec<_>>();

            let mut available: Vec<(u32, BinaryHeap<AvailablePass>)> =
                vec![(0, BinaryHeap::new()); self.queues.len()];
            let mut scheduled: Vec<GraphPassEvent> = Vec::new();

            let mut graph_layers = vec![0; self.passes.len()];
            // in a bfs, each node gets a "layer" in which is the maximum distance from a root node
            // we would like to use this in the scheduling heuristic because there are no dependencies within each layer
            // and we can saturate the gpu better
            for (pass_i, &(dep_count, _)) in dependency_count.iter().enumerate() {
                let p = GraphPass::new(pass_i);
                if !self.is_pass_alive(p) {
                    graph_layers[pass_i] = -1;
                    continue;
                }
                // root nodes get a layer 1 and are pushed into the available heaps
                if dep_count == 0 {
                    graph_layers[pass_i] = 1;
                }
            }

            let mut queue_flush_region = vec![0usize; self.queues.len()];

            // currently we are creating the scheduled passes by looping over each queue and poppping the locally optimal pass
            // this is rather questionable so TODO think this over
            loop {
                let len = scheduled.len();
                for queue_i in 0..self.queues.len() {
                    let (position, heap) = &mut available[queue_i];

                    let pass;
                    if let Some(AvailablePass { pass: p, .. }) = heap.pop() {
                        pass = p;
                    } else {
                        // we've depleted the last flush region, continue to the next one
                        let index = &mut queue_flush_region[queue_i];
                        for &e in &self.timeline[*index..] {
                            *index += 1;
                            match e.get() {
                                PassEventData::Pass(next_pass) => {
                                    let data = &self.get_pass_data(next_pass);
                                    if data.queue.index() != queue_i {
                                        continue;
                                    }
                                    let dependency_info = &mut dependency_count[next_pass.index()];
                                    // if the pass has no outstanding deendencies, we measure its priority and add it to the heap
                                    if dependency_info.0 == 0 {
                                        let queue = data.queue;
                                        let item =
                                            AvailablePass::new(next_pass, self, &mut graph_layers);
                                        heap.push(item);
                                    }
                                    dependency_info.1 = true;
                                }
                                PassEventData::Flush(f) => {
                                    if f.index() == queue_i {
                                        scheduled.push(GraphPassEvent::new(PassEventData::Flush(
                                            GraphQueue::new(queue_i),
                                        )));
                                        break;
                                    }
                                }
                                // moves are handled later
                                PassEventData::Move(_) => {}
                            }
                        }

                        if let Some(AvailablePass { pass: p, .. }) = heap.pop() {
                            pass = p;
                        } else {
                            continue;
                        }
                    };

                    scheduled.push(GraphPassEvent::new(PassEventData::Pass(pass)));
                    *position += 1;

                    for &child in self.get_children(pass) {
                        if !self.is_pass_alive(child) {
                            continue;
                        }

                        let count = &mut dependency_count[child.index()];
                        count.0 -= 1;
                        if count.0 == 0 && count.1 == true {
                            let queue = self.get_pass_data(child).queue;
                            let item = AvailablePass::new(child, self, &mut graph_layers);
                            available[queue.index()].1.push(item);
                        }
                    }
                }

                // if the length is unchanged we have exhausted all passes
                if len == scheduled.len() {
                    break;
                }
            }
            scheduled
        };

        // collect live intervals of the resources
        // this is used for physical resource physical resource assignment and correct image move handling
        let (image_usage, buffer_usage) = {
            let mut image_usage = vec![0..0; self.images.len()];
            let mut buffer_usage = vec![0..0; self.buffers.len()];

            for (event_i, e) in scheduled.iter().enumerate() {
                match e.get() {
                    PassEventData::Pass(p) => {
                        let data = self.get_pass_data(p);
                        for i in &data.images {
                            let usage = &mut image_usage[i.handle.index()];
                            if usage.start == 0 {
                                usage.start = event_i;
                            }
                            usage.end = event_i;
                        }
                        for b in &data.buffers {
                            let usage = &mut buffer_usage[b.handle.index()];
                            if usage.start == 0 {
                                usage.start = event_i;
                            }
                            usage.end = event_i;
                        }
                    }
                    PassEventData::Move(_) => {}
                    PassEventData::Flush(_) => {}
                }
            }
            (image_usage, buffer_usage)
        };

        // the moment the move-src images aren't used anymore, they are "moved"
        // Vec<(index of pass after which the move occurs, handle to the move to perform)>
        let mut image_moves = self
            .moves
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let pass = TimelinePass::new(
                    m.from
                        .iter()
                        .map(|i| image_usage[i.index()].end)
                        .max()
                        .unwrap(),
                );
                let mov = GraphPassMove::new(i);
                (pass, mov)
            })
            .collect::<Vec<_>>();
        image_moves.sort_unstable_by_key(|(p, _)| *p);

        // separate the scheduled passes into specific submissions
        // this is a greedy process, but due to the previous scheduling it should yield somewhat decent results
        let (submissions, image_last_state, buffer_last_state, accessors) = {
            let mut image_rw: Vec<(ResourceState<ImageMarker>, bool)> =
                vec![(ResourceState::default(), false); self.images.len()];
            let mut buffer_rw: Vec<(ResourceState<BufferMarker>, bool)> =
                vec![(ResourceState::default(), false); self.buffers.len()];

            for i in 0..self.images.len() {
                let image = GraphImage::new(i);
                image_rw[i].1 = self.get_image_data(image).is_sharing_concurrent();
            }

            for i in 0..self.buffers.len() {
                let buffer = GraphBuffer::new(i);
                buffer_rw[i].1 = self.get_buffer_data(buffer).is_sharing_concurrent();
            }

            let mut recorder = RefCell::new(SubmissionRecorder::new(self));

            // some random buffers extracted here to keep the allocations
            let mut waitfor_submissions: Vec<GraphSubmission> = Vec::new();
            let mut waitfor_resources: Vec<GraphResource> = Vec::new();
            // (src queue family, src pass, dst pass, resource in question)
            let mut queued_ownership_transitions: SmallVec<
                [(
                    u32,
                    Vec<(
                        GraphSubmission,
                        SubmissionPass,
                        SubmissionPass,
                        GraphResource,
                    )>,
                ); 8],
            > = self
                .get_queue_families()
                .iter()
                .map(|f| (*f, Vec::new()))
                .collect();

            // set of accesses is the first that occurs in the timeline
            // becomes "closed" (bool = true) when the first write occurs
            // this will be used during graph execution to synchronize with external resources
            let mut accessors: HashMap<GraphResource, ResourceFirstAccess> =
                constant_ahash_hashmap();

            for (timeline_i, &e) in scheduled.iter().enumerate() {
                match e.get() {
                    PassEventData::Pass(p) => {
                        let on_end = |recorder: &mut SubmissionRecorder| {
                            let dst_queue_family =
                                self.get_queue_family(recorder.get_current_submission().queue);

                            for &mut (src_queue_family, ref mut src_submissions) in
                                &mut queued_ownership_transitions
                            {
                                use slice_group_by::GroupBy;
                                // list of submissions which we need to wait on to perform a queue ownership release
                                waitfor_submissions.clear();
                                // list of resources that will be released and then acquired
                                waitfor_resources.clear();

                                src_submissions.sort_unstable_by_key(|(_, _, _, s)| *s);
                                for entries in
                                    src_submissions.binary_group_by_key(|(_, _, _, s)| *s)
                                {
                                    // assert that only one dst pass is causing the transition
                                    debug_assert!(entries[1..]
                                        .iter()
                                        .all(|(_, _, dst, _)| *dst == entries[0].2));

                                    // the current submission gets acquires the ownership for the resources
                                    for &(submission, src_pass, dst_pass, data) in entries {
                                        let barrier = match data {
                                            GraphResource::Image(image) => {
                                                SpecialBarrier::ImageOwnershipTransition {
                                                    image,
                                                    src_family: src_queue_family,
                                                    dst_family: dst_queue_family,
                                                }
                                            }
                                            GraphResource::Buffer(buffer) => {
                                                SpecialBarrier::BufferOwnershipTransition {
                                                    buffer,
                                                    src_family: src_queue_family,
                                                    dst_family: dst_queue_family,
                                                }
                                            }
                                        };
                                        recorder
                                            .get_current_submission_mut()
                                            .special_barriers
                                            .push((dst_pass, barrier));
                                    }

                                    // emit the ownership releases
                                    match entries {
                                        &[] => unreachable!(),
                                        // only a single dependency, put the barrier straigh after the src pass
                                        &[(submission, src_pass, dst_pass, data)] => {
                                            let sub =
                                                recorder.get_closed_submission_mut(submission);
                                            // hack the handle to point to the pass after the src
                                            // FIXME maybe just handle ownership release differently from other barriers
                                            let pass_after =
                                                SubmissionPass(src_pass.0.checked_add(1).unwrap());
                                            let barrier = match data {
                                                GraphResource::Image(image) => {
                                                    SpecialBarrier::ImageOwnershipTransition {
                                                        image,
                                                        src_family: src_queue_family,
                                                        dst_family: dst_queue_family,
                                                    }
                                                }
                                                GraphResource::Buffer(buffer) => {
                                                    SpecialBarrier::BufferOwnershipTransition {
                                                        buffer,
                                                        src_family: src_queue_family,
                                                        dst_family: dst_queue_family,
                                                    }
                                                }
                                            };
                                            sub.special_barriers.push((pass_after, barrier));
                                        }
                                        &[(submission, _, _, data), ..] => {
                                            // if all of the accesses are within the same submission, we can just add a dummy pass at the end
                                            // which waits for all of the passes and then releases ownership
                                            if entries[1..].iter().all(|(s, ..)| *s == entries[0].0)
                                            {
                                                let sub =
                                                    recorder.get_closed_submission_mut(submission);

                                                // barriers that don't target actual passes will require special handling during execution
                                                let dummy_pass =
                                                    SubmissionPass::new(sub.passes.len());

                                                // FIXME check whether such a barrier is already in place
                                                sub.barriers.push((dummy_pass, SimpleBarrier {
                                                    // TODO accumulate actual accesses of passes and use that 
                                                    src_stages: vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                                                    src_access: vk::AccessFlags2KHR::all(),
                                                    dst_stages: vk::PipelineStageFlags2KHR::empty(),
                                                    dst_access: vk::AccessFlags2KHR::empty(),
                                                }));
                                                let barrier = match data {
                                                    GraphResource::Image(image) => {
                                                        SpecialBarrier::ImageOwnershipTransition {
                                                            image,
                                                            src_family: src_queue_family,
                                                            dst_family: dst_queue_family,
                                                        }
                                                    }
                                                    GraphResource::Buffer(buffer) => {
                                                        SpecialBarrier::BufferOwnershipTransition {
                                                            buffer,
                                                            src_family: src_queue_family,
                                                            dst_family: dst_queue_family,
                                                        }
                                                    }
                                                };
                                                sub.special_barriers.push((dummy_pass, barrier));
                                            } else {
                                                // we need to create a dummy submission that synchronizes all of the previous ones
                                                // this is done later, for now only push the required data
                                                for &(submission, _, _, data) in entries {
                                                    if !waitfor_submissions.contains(&submission) {
                                                        waitfor_submissions.push(submission);
                                                    }
                                                    // FIXME this check may be unneccessary
                                                    if !waitfor_resources.contains(&data) {
                                                        waitfor_resources.push(data);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // here is the dummy submission stuff
                                if !waitfor_submissions.is_empty() {
                                    // find a queue that has the src family
                                    let queue =
                                        recorder.find_queue_with_family(src_queue_family).unwrap();

                                    let barriers = waitfor_resources.iter().map(|&res| {
                                        (
                                            SubmissionPass(0),
                                            match res {
                                                GraphResource::Image(image) => {
                                                    SpecialBarrier::ImageOwnershipTransition {
                                                        image,
                                                        src_family: src_queue_family,
                                                        dst_family: dst_queue_family,
                                                    }
                                                }
                                                GraphResource::Buffer(buffer) => {
                                                    SpecialBarrier::BufferOwnershipTransition {
                                                        buffer,
                                                        src_family: src_queue_family,
                                                        dst_family: dst_queue_family,
                                                    }
                                                }
                                            },
                                        )
                                    });

                                    let submission = Submission {
                                        queue,
                                        passes: Default::default(),
                                        semaphore_dependencies: waitfor_submissions.clone(),
                                        barriers: Default::default(),
                                        special_barriers: barriers.collect(),
                                    };

                                    let sub = recorder.add_submission_sneaky(submission);
                                    recorder.add_semaphore_dependency(sub);
                                }
                            }
                            // clear the data we've just processed
                            for v in &mut queued_ownership_transitions {
                                v.1.clear();
                            }
                        };
                        let timeline_pass = e.to_timeline_pass().unwrap();
                        let submission_pass =
                            recorder.borrow_mut().begin_pass(timeline_pass, p, on_end);

                        let data = self.get_pass_data(p);
                        let queue = data.queue;
                        let queue_family = self.get_queue_family(queue);

                        for img in &data.images {
                            self.handle_resource(
                                p,
                                timeline_pass,
                                submission_pass,
                                queue_family,
                                img,
                                &recorder,
                                &mut image_rw,
                                &mut queued_ownership_transitions,
                                &mut accessors,
                            );
                        }

                        for buf in &data.buffers {
                            self.handle_resource(
                                p,
                                timeline_pass,
                                submission_pass,
                                queue_family,
                                buf,
                                &recorder,
                                &mut buffer_rw,
                                &mut queued_ownership_transitions,
                                &mut accessors,
                            );
                        }

                        let mut recorder_ref = recorder.borrow_mut();
                        for dep in &data.dependencies {
                            if dep.is_real() {
                                recorder_ref.add_dependency(
                                    dep.get_pass(),
                                    vk::AccessFlags2KHR::all(),
                                    vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                                    data.access,
                                    data.stages,
                                )
                            }
                        }

                        // apply image moves
                        for (_, mov) in image_moves
                            .iter()
                            .take_while(|(move_pass, _)| move_pass.index() <= timeline_i)
                        {
                            let data = &self.moves[mov.index()];

                            let mut new_parts: SmallVec<
                                [(PassTouch, TypeSome<vk::ImageLayout>, GraphImage); 4],
                            > = SmallVec::new();

                            for &i in &data.from {
                                match std::mem::replace(
                                    &mut image_rw[i.index()].0,
                                    ResourceState::Moved,
                                ) {
                                    ResourceState::MoveDst { parts } => {
                                        new_parts.extend(parts.unwrap())
                                    }
                                    ResourceState::Normal {
                                        layout,
                                        queue_family,
                                        access,
                                    } => {
                                        // TODO verify this, seems finicky?
                                        for touch in access {
                                            new_parts.push((touch, layout, i));
                                        }
                                    }
                                    _ => panic!("Resource in unsupported state"),
                                }
                            }
                            let state = &mut image_rw[data.to.index()].0;
                            let ResourceState::Uninit = *state else {
                                panic!("Image move destination must be unitialized!");
                            };
                            *state = ResourceState::<ImageMarker>::MoveDst {
                                parts: TypeSome::new_some(new_parts),
                            };
                        }
                    }
                    PassEventData::Move(_) => {} // this is handled differently
                    PassEventData::Flush(q) => recorder.borrow_mut().close_current_submission(),
                }
            }

            let mut submissions = RefCell::into_inner(recorder).finish();

            for sub in &mut submissions {
                // may not be sorted due to later submissions possibly adding barriers willy nilly
                sub.barriers.sort_unstable_by_key(|(pass, _)| *pass);
                sub.special_barriers
                    .sort_unstable_by_key(|(pass, barrier)| {
                        // FIXME kinda stupid
                        let resource = match barrier {
                            &SpecialBarrier::LayoutTransition { image, .. } => {
                                GraphResource::Image(image)
                            }
                            &SpecialBarrier::ImageOwnershipTransition { image, .. } => {
                                GraphResource::Image(image)
                            }
                            &SpecialBarrier::BufferOwnershipTransition { buffer, .. } => {
                                GraphResource::Buffer(buffer)
                            }
                        };
                        (*pass, resource)
                    });
            }

            // perform transitive reduction on the submissions
            // lifted from petgraph, https://docs.rs/petgraph/latest/petgraph/algo/tred/fn.dag_transitive_reduction_closure.html
            {
                // make sure that the dependencies are in topological order (ie are sorted)
                for sub in &mut submissions {
                    sub.semaphore_dependencies.sort_unstable();
                }

                let len = submissions.len();

                let mut tred = vec![Vec::new(); len];
                let mut tclos = Vec::with_capacity(len);
                let mut mark = vec![false; len];

                for i in 0..len {
                    tclos.push(Vec::with_capacity(
                        submissions[i].semaphore_dependencies.len(),
                    ));
                }

                // since each node has a list of predecessors and not successors, we need to reverse the order of all iteration
                // in relation to the petgraph implementation, it does actually work
                for i in 0..len {
                    for &x in submissions[i].semaphore_dependencies.iter().rev() {
                        if !mark[x.index()] {
                            tred[i].push(x);
                            tclos[i].push(x);

                            for y_i in (0..tclos[x.index()].len()).rev() {
                                let y = tclos[x.index()][y_i];
                                if !mark[y.index()] {
                                    mark[y.index()] = true;
                                    tclos[i].push(y);
                                }
                            }
                        }
                    }
                    for y in &tclos[i] {
                        mark[y.index()] = false;
                    }
                }

                for (submission, reduced_dependencies) in
                    submissions.iter_mut().zip(tred.into_iter())
                {
                    submission.semaphore_dependencies = reduced_dependencies;
                }
            }

            (submissions, image_rw, buffer_rw, accessors)
        };

        // now we start assigning physical resources to those we've just created
        // we only own the backing memory of transient resources, so we only need to consider those
        // moved source-resources do not count

        // the ResourceState objects should track the last relevant access to the resources, after a pass
        // has synchronized with all of these accesses, we can safely reuse their memory

        // the assignment will have multiple phases:
        // 1. try to find a dead resource with a compatible format/extent/usage...
        // 2. pick a piece of free memory (including that which backs dead resources), remove the dead resource completely, and use its memory
        // 3. if we've exhausted some memory limit, abort, free all resources, and allocate linearly without fragmentation
        //    TODO do better than a hungry allocation strategy, simulated annealing? backtracking? async bruteforce?

        let deferred_free = {
            let mut submission_children = ImmutableGraph::new();
            let graph = SubmissionFacade(&submissions, ());
            reverse_edges_into(&graph, &mut submission_children);

            let mut scratch = Vec::new();

            let mut submission_reuse = vec![SubmissionResourceReuse::default(); submissions.len()];
            // TODO profile how many children submissions usually have
            // a vector may be faster here
            let mut intersection_hashmap: ahash::HashMap<GraphSubmission, (GraphSubmission, u32)> =
                constant_ahash_hashmap();
            let mut availibility_interner: (
                AvailabilityToken,
                ahash::HashMap<ResourceLastUse, AvailabilityToken>,
            ) = (AvailabilityToken::new(), constant_ahash_hashmap());

            let mut image_available = vec![AvailabilityToken::NONE; self.images.len()];
            let image_last_touch = self.get_last_resource_usage(&image_last_state, &mut scratch);
            submission_fill_reuse::<ImageMarker>(
                image_last_touch,
                &mut image_available,
                &mut submission_reuse,
                &mut intersection_hashmap,
                &submission_children,
                &mut availibility_interner,
            );
            let mut buffer_available = vec![AvailabilityToken::NONE; self.buffers.len()];
            let buffer_last_touch = self.get_last_resource_usage(&buffer_last_state, &mut scratch);
            submission_fill_reuse::<BufferMarker>(
                buffer_last_touch,
                &mut buffer_available,
                &mut submission_reuse,
                &mut intersection_hashmap,
                &submission_children,
                &mut availibility_interner,
            );

            for a in self.memory.borrow_mut().iter_mut() {
                a.allocator.reset();
            }

            // Resource reuse is currently not implemented since we will need a more extensive liveness tracking scheme
            // the current idea is to represent the lifetime of each resource as two "edges" - two sets of nodes of all the passes which use the resource
            // that serve as the roots of the graph:
            //
            //  [X] [X] [X]     start_edge_nodes marked with []
            //    \ /   /
            //     X   X
            //     |\ / \
            //     | X   \
            //     |/     \
            //    (X)     (X)   end_edge_nodes with ()
            //
            // We will also precompute all passes (use submissions for external passes instead) which are "observed" to end or or which can observe end edge nodes
            // on a simpler case:
            //
            //   S   S    -- start_nodes which all [X] passes can observe to end
            //    \ /
            //    [X]     -- start_edge_nodes
            //     |
            //     X
            //     |
            //    (X)     -- end_edge_nodes
            //    / \
            //   E   E    -- end_nodes which can observe all (X) passes end
            //
            // When checking where two ranges are disjoint and safe to alias, we'll return whether end_nodes are all observable from start_edge_nodes
            // we will perform this check two times for both permutations of the two ranges, if none return true, the ranges aren't disjoint
            // as an optimization, we seemingly only need to store end_nodes and start_edge_nodes to do this.
            //
            // We will also need to actually build a dependency graph for passes within a submission, we cannot reuse the existing one since the use of barriers
            // may create more dependencies and it would be useful to track / exploit them.

            // macro_rules! generate_hash_eq {
            //     ($hash_fun:ident, $cmp_fun:ident, ($state_ty1:ty, $state_ty2:ty); $($field1:ident)+ + $($field2:ident)*) => {
            //         fn $hash_fun(data1: $state_ty1, data2: $state_ty2) -> u64 {
            //             let mut hasher = constant_ahash_hasher();
            //             $(
            //                 data1.$field1.hash(&mut hasher);
            //             )+
            //             $(
            //                 data2.$field2.hash(&mut hasher);
            //             )*
            //             hasher.finish()
            //         }
            //         fn $cmp_fun((data1_1, data2_1): ($state_ty1, $state_ty2), (data1_2, data2_2): ($state_ty1, $state_ty2)) -> bool {
            //             $(
            //                 data1_1.$field1 == data1_2.$field1
            //             )&&+
            //             $(
            //                 && data2_1.$field2 == data2_2.$field2
            //             )*
            //         }
            //     };
            // }

            // struct Dummy<T> {
            //     i: T,
            // }

            // impl<T> Dummy<T> {
            //     fn new(i: T) -> Self {
            //         Self { i }
            //     }
            // }

            // // image reuse

            // generate_hash_eq! {
            //     image_info_hash, image_info_eq, (&ImageCreateInfo, Dummy<u32>);
            //     flags
            //     size
            //     format
            //     samples
            //     mip_levels
            //     array_layers
            //     tiling
            //     usage
            //     sharing_mode_concurrent
            //     +
            //     i
            // }

            let mut physical_images = self.physical_images.drain(..).map(Some).collect::<Vec<_>>();

            // let mut image_reuse = self
            //     .physical_images
            //     .iter()
            //     .enumerate()
            //     .map(|(i, data)| {
            //         let hash = image_info_hash(&data.info, Dummy::new(data.get_memory_type()));
            //         let handle = PhysicalImage::new(i);
            //         (hash, handle)
            //     })
            //     .collect::<Vec<_>>();

            // image_reuse.sort_by_key(|&(hash, _)| hash);

            // // first try to straight out reuse previously created resources
            // for (i, img) in self.images.iter_mut().enumerate() {
            //     let meta = &self.image_meta[i];
            //     if !meta.alive.get() {
            //         continue;
            //     }
            //     let transient = GraphObject::deref_mut(img);
            //     match transient {
            //         ImageData::TransientPrototype(info, allocation_info) => {
            //             let allocator = self.device.allocator();
            //             let memory_type = unsafe {
            //                 allocator
            //                     .find_memory_type_index(!0, allocation_info)
            //                     .unwrap()
            //             };

            //             let hash = image_info_hash(info, Dummy::new(memory_type));

            //             let Ok(found) = image_reuse.binary_search_by_key(&hash, |&(hash, _)| hash) else {
            //                 continue;
            //             };

            //             let (_, physical_image) = image_reuse[found];
            //             let physical_data = &self.physical_images[physical_image.index()];

            //             if image_info_eq(
            //                 (info, Dummy::new(memory_type)),
            //                 (
            //                     &physical_data.info,
            //                     Dummy::new(physical_data.get_memory_type()),
            //                 ),
            //             ) {
            //                 // TODO check that we can alias the memory

            //                 // FIXME use some structure with more efficient removes
            //                 image_reuse.remove(found);

            //                 let handle = PhysicalImage::new(self.physical_images.len());
            //                 let data = physical_images[physical_image.index()].take().unwrap();
            //                 self.physical_images.push(data);
            //                 *transient = ImageData::Transient(handle);
            //             }
            //         }
            //         ImageData::Transient(_) => {
            //             unreachable!("Transient resources are just getting assigned")
            //         }
            //         ImageData::Imported(_) => {}
            //         ImageData::Swapchain(_) => {}
            //         ImageData::Moved(_, _, _) => {}
            //     }
            // }

            // // buffer reuse

            // generate_hash_eq! {
            //     buffer_info_hash, buffer_info_eq, (Dummy<&BufferCreateInfo>, Dummy<u32>);
            //     i
            //     +
            //     i
            // }

            let mut physical_buffers = self
                .physical_buffers
                .drain(..)
                .map(Some)
                .collect::<Vec<_>>();

            // let mut buffer_reuse = self
            //     .physical_buffers
            //     .iter()
            //     .enumerate()
            //     .map(|(i, data)| {
            //         let hash = buffer_info_hash(
            //             Dummy::new(&data.info),
            //             Dummy::new(data.get_memory_type()),
            //         );
            //         let handle = PhysicalBuffer::new(i);
            //         (hash, handle)
            //     })
            //     .collect::<Vec<_>>();

            // buffer_reuse.sort_by_key(|&(hash, _)| hash);

            // // first try to straight out reuse previously created resources
            // // FIXME code duplication
            // for (i, img) in self.buffers.iter_mut().enumerate() {
            //     let meta = &self.buffer_meta[i];
            //     if !meta.alive.get() {
            //         continue;
            //     }
            //     let transient = GraphObject::deref_mut(img);
            //     match transient {
            //         BufferData::TransientPrototype(info, allocation_info) => {
            //             let allocator = self.device.allocator();
            //             let memory_type = unsafe {
            //                 allocator
            //                     .find_memory_type_index(!0, allocation_info)
            //                     .unwrap()
            //             };

            //             let hash = buffer_info_hash(Dummy::new(info), Dummy::new(memory_type));

            //             let Ok(found) = buffer_reuse.binary_search_by_key(&hash, |&(hash, _)| hash) else {
            //                 continue;
            //             };

            //             let (_, physical_buffer) = buffer_reuse[found];
            //             let physical_data = &self.physical_buffers[physical_buffer.index()];

            //             if buffer_info_eq(
            //                 (Dummy::new(info), Dummy::new(memory_type)),
            //                 (
            //                     Dummy::new(&physical_data.info),
            //                     Dummy::new(physical_data.get_memory_type()),
            //                 ),
            //             ) {
            //                 // FIXME use some structure with more efficient removes
            //                 buffer_reuse.remove(found);

            //                 let handle = PhysicalBuffer::new(self.physical_buffers.len());
            //                 let data = physical_buffers[physical_buffer.index()].take().unwrap();
            //                 self.physical_buffers.push(data);
            //                 *transient = BufferData::Transient(handle);
            //             }
            //         }
            //         BufferData::Transient(_) => {
            //             unreachable!("Transient resources are just getting assigned")
            //         }
            //         BufferData::Imported(_) => {}
            //     }
            // }

            let mut deferred_free = Vec::new();

            // schedule the remaining images to be freed, we can't do this now since we've not waited for the possibly pending submissions from the previous graph execution to complete
            // when aliasing memory, we will consider them unused as command buffers are filled only after such synchronization has happened
            for img in physical_images.into_iter().flatten() {
                let memory_type = img.get_memory_type();
                let PhysicalImageData {
                    info,
                    memory,
                    vkhandle,
                    state,
                } = img;
                self.get_suballocator(memory_type);
                deferred_free.push(DeferredResourceFree::Image { vkhandle, state });
            }

            // free the remaining buffers
            for buf in physical_buffers.into_iter().flatten() {
                let memory_type = buf.get_memory_type();
                let PhysicalBufferData {
                    info,
                    memory,
                    vkhandle,
                    state,
                } = buf;
                self.get_suballocator(memory_type);
                deferred_free.push(DeferredResourceFree::Buffer { vkhandle, state });
            }

            #[derive(Clone, Copy)]
            enum NeedAlloc {
                Image {
                    vkhandle: vk::Image,
                    handle: GraphImage,
                },
                Buffer {
                    vkhandle: vk::Buffer,
                    handle: GraphBuffer,
                },
            }

            let mut need_alloc = Vec::new();
            let mut pass_effects: Vec<(vk::PipelineStageFlags2KHR, Vec<AvailabilityToken>)> =
                Vec::new();
            // token, refcount
            let mut after_local_passes: ahash::HashMap<AvailabilityToken, u32> =
                constant_ahash_hashmap();
            for (i, submission) in submissions.iter().enumerate() {
                SubmissionResourceReuse::begin(
                    GraphSubmission::new(i),
                    &submissions,
                    &mut submission_reuse,
                );

                if submission.passes.len() == 1 {
                    let a = 2;
                }

                if submission.passes.len() > pass_effects.len() {
                    pass_effects.resize(submission.passes.len(), Default::default());
                }

                for (_, vec) in &mut pass_effects {
                    vec.clear();
                }

                let reuse = &mut submission_reuse[i];
                for &(token, ref passes) in &reuse.after_local_passes {
                    for p in passes {
                        pass_effects[p.index()].1.push(token);
                    }
                    after_local_passes.insert(token, passes.len() as u32);
                }

                let mut barriers = submission.barriers.iter();
                for (i, &p) in submission.passes.iter().enumerate() {
                    let data = self.get_pass_data(p);

                    pass_effects[i].0 = data.stages;

                    let mut dst_stages = vk::PipelineStageFlags2KHR::empty();
                    while let Some((pass, barrier)) = barriers.clone().next() {
                        if pass.index() == i {
                            dst_stages |= barrier.dst_stages;
                            barriers.next();
                        } else {
                            break;
                        }
                    }
                    dst_stages = dst_stages.translate_special_bits();

                    if !dst_stages.is_empty() {
                        for j in 0..i {
                            let (effects, vec) = &mut pass_effects[j];
                            if !effects.is_empty() {
                                *effects &= !dst_stages;
                                if effects.is_empty() {
                                    for token in vec {
                                        match after_local_passes.entry(*token) {
                                            Entry::Occupied(mut occupied) => {
                                                let refcount = occupied.get_mut();
                                                *refcount -= 1;
                                                if *refcount == 0 {
                                                    reuse
                                                        .current_available_intervals
                                                        .insert(*token);
                                                    occupied.remove();
                                                }
                                            }
                                            Entry::Vacant(_) => panic!(),
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let device = self.device.device();

                    need_alloc.clear();
                    for a in &data.images {
                        if let ImageData::TransientPrototype(create_info, allocation_info) =
                            self.get_image_data(a.handle)
                        {
                            let vk_info = create_info.to_vk();
                            let image = unsafe {
                                device
                                    .create_image(&vk_info, self.device.allocator_callbacks())
                                    .unwrap()
                            };
                            let requirements =
                                unsafe { device.get_image_memory_requirements(image) };
                            need_alloc.push((
                                NeedAlloc::Image {
                                    vkhandle: image,
                                    handle: a.handle,
                                },
                                requirements,
                            ));
                        }
                    }
                    for a in &data.buffers {
                        if let BufferData::TransientPrototype(create_info, allocation_info) =
                            self.get_buffer_data(a.handle)
                        {
                            let vk_info = create_info.to_vk();
                            let buffer = unsafe {
                                device
                                    .create_buffer(&vk_info, self.device.allocator_callbacks())
                                    .unwrap()
                            };
                            let requirements =
                                unsafe { device.get_buffer_memory_requirements(buffer) };
                            need_alloc.push((
                                NeedAlloc::Buffer {
                                    vkhandle: buffer,
                                    handle: a.handle,
                                },
                                requirements,
                            ));
                        }
                    }

                    // first deal with the largest resources
                    need_alloc.sort_unstable_by(|(_, a), (_, b)| a.size.cmp(&b.size).reverse());

                    for (resource, reqs) in &need_alloc {
                        let (tiling, allocation_info, availability, display) = match *resource {
                            NeedAlloc::Image { vkhandle, handle } => {
                                let ImageData::TransientPrototype(create_info, allocation_info) = self.get_image_data(handle) else {
                                    unreachable!()
                                };
                                (
                                    create_info.tiling.into(),
                                    allocation_info,
                                    image_available[handle.index()],
                                    self.get_image_display(handle),
                                )
                            }
                            NeedAlloc::Buffer { vkhandle, handle } => {
                                let BufferData::TransientPrototype(create_info, allocation_info) = self.get_buffer_data(handle) else {
                                    unreachable!()
                                };
                                (
                                    MemoryKind::Linear,
                                    allocation_info,
                                    buffer_available[handle.index()],
                                    self.get_buffer_display(handle),
                                )
                            }
                        };

                        let allocator = self.device.allocator();
                        let buffer_image_granularity = self
                            .device
                            .physical_device_properties
                            .limits
                            .buffer_image_granularity;

                        let memory_type = unsafe {
                            allocator
                                .find_memory_type_index(reqs.memory_type_bits, allocation_info)
                                .unwrap_or_else(|_| {
                                    panic!(
                                        "Failed to find memory_type for resource \"{}\"",
                                        display
                                    )
                                })
                        };
                        let suballocation = self
                            .get_suballocator(memory_type)
                            .allocate(
                                reqs.size,
                                reqs.alignment,
                                tiling,
                                availability,
                                buffer_image_granularity,
                                memory_type,
                                &reuse,
                                || unsafe { allocator.allocate_memory(reqs, allocation_info) },
                            )
                            // TODO oom handling
                            .expect("Failed to allocate memory from VMA, TODO handle this");

                        let allocation = suballocation.memory.allocation;
                        let memory_info =
                            unsafe { self.device.allocator().get_allocation_info(allocation) };

                        match *resource {
                            NeedAlloc::Image { vkhandle, handle } => {
                                let data = self.get_image_data(handle);
                                let ImageData::TransientPrototype(create_info, allocation_info) = data else {
                                    unreachable!()
                                };
                                let allocation = suballocation.memory.allocation;

                                unsafe {
                                    // TODO consider using bind_image_memory2
                                    self.device
                                        .device()
                                        .bind_image_memory(
                                            vkhandle,
                                            memory_info.device_memory,
                                            memory_info.offset + suballocation.offset,
                                        )
                                        .unwrap();
                                }
                                let physical = PhysicalImage::new(self.physical_images.len());
                                self.physical_images.push(PhysicalImageData {
                                    info: create_info.clone(),
                                    memory: suballocation,
                                    vkhandle,
                                    state: ResourceMutableState::with_initial_layout(
                                        vk::ImageLayout::UNDEFINED,
                                    ),
                                });
                                *self.get_image_data_mut(handle) = ImageData::Transient(physical);
                            }
                            NeedAlloc::Buffer { vkhandle, handle } => {
                                let data = self.get_buffer_data(handle);
                                let BufferData::TransientPrototype(create_info, allocation_info) = data else {
                                    unreachable!()
                                };
                                let allocation = suballocation.memory.allocation;

                                unsafe {
                                    // TODO consider using bind_buffer_memory2
                                    self.device
                                        .device()
                                        .bind_buffer_memory(
                                            vkhandle,
                                            memory_info.device_memory,
                                            memory_info.offset + suballocation.offset,
                                        )
                                        .unwrap();
                                }
                                let physical = PhysicalBuffer::new(self.physical_buffers.len());
                                self.physical_buffers.push(PhysicalBufferData {
                                    info: create_info.clone(),
                                    memory: suballocation,
                                    vkhandle,
                                    state: ResourceMutableState::new(),
                                });
                                *self.get_buffer_data_mut(handle) = BufferData::Transient(physical);
                            }
                        }
                    }
                }

                // flush all the local-pass availability
                for (token, refcount) in after_local_passes.drain() {
                    assert!(refcount > 0, "refcounts that reach 0 should be removed");
                    reuse.current_available_intervals.insert(token);
                }
            }

            deferred_free
        };

        // we can now start executing the graph (TODO move this to another function, allow the built graph to be executed multiple times)
        // wait for previously submitted work to finish
        if let Some(id) = self.prev_generation.get() {
            self.device
                .wait_for_generation_single(id, u64::MAX)
                .unwrap();
        }

        for (pass, meta) in self.passes.iter_mut().zip(&self.pass_meta) {
            if meta.alive.get() {
                pass.pass.borrow_mut().prepare();
            }
        }

        // since we've now waited for all previous work to finish, we can safely reset the semaphores
        unsafe {
            self.swapchain_semaphores.reset();
            self.device
                .device()
                .reset_command_pool(self.command_pool, None);
        }

        for pending in deferred_free {
            unsafe {
                match pending {
                    DeferredResourceFree::Image { vkhandle, state } => {
                        state.destroy(&self.device);
                        self.device
                            .device()
                            .destroy_image(vkhandle, self.device.allocator_callbacks());
                    }
                    DeferredResourceFree::Buffer { vkhandle, state } => {
                        state.destroy(&self.device);
                        self.device
                            .device()
                            .destroy_buffer(vkhandle, self.device.allocator_callbacks());
                    }
                }
            }
        }

        // each submission gets a semaphore
        // TODO use sequential semaphore allocation to improve efficiency
        let semaphores = submissions
            .iter()
            .map(|_| self.device.make_submission())
            .collect::<Vec<_>>();

        // TODO do something about this so that we don't hold the lock for all of execution
        let image_storage_lock = self.device.image_storage.acquire_all_exclusive();
        let buffer_storage_lock = self.device.buffer_storage.acquire_all_exclusive();
        let swapchain_storage_lock = self.device.swapchain_storage.acquire_all_exclusive();

        struct SwapchainPreset {
            swapchain: GraphImage,
            vkhandle: vk::SwapchainKHR,
            image: vk::Image,
            image_index: u32,
            image_acquire: vk::Semaphore,
            image_release: vk::Semaphore,
        }

        let mut semaphore_scratch = Vec::new();
        let mut waited_idle = false;
        let mut swapchain_indices: ahash::HashMap<GraphImage, u32> = constant_ahash_hashmap();
        let mut submission_swapchains: ahash::HashMap<QueueSubmission, Vec<SwapchainPreset>> =
            constant_ahash_hashmap();

        // collect the first accesses to all external resources, we will need to synchronize against them and possibly perform queue family and layout transitions

        for (i, img) in self.images.iter().enumerate() {
            let handle = GraphImage::new(i);
            let data = self.get_image_meta(handle);

            if data.alive.get() == false {
                continue;
            }

            match img.deref() {
                ImageData::Imported(_) | ImageData::Swapchain(_) => {
                    let resource = GraphResource::Image(handle);
                    let usage = accessors.get(&resource).unwrap();

                    let ResourceState::Normal { layout, queue_family, access } = &image_last_state[i].0 else {
                        panic!("Unsupported resource state (TODO should Uninit be allowed?)");
                    };

                    semaphore_scratch.clear();
                    semaphore_scratch.extend(usage.accessors.iter().map(|a| {
                        let pass = self.timeline[a.index()].get_pass().unwrap();
                        let submission =
                            self.get_pass_meta(pass).scheduled_submission.get().unwrap();
                        semaphores[submission as usize].0
                    }));

                    // nothing is using the resource, continue
                    // TODO is this even possible? wouldn't the resource be dead?
                    if semaphore_scratch.len() == 0 {
                        continue;
                    }

                    semaphore_scratch.sort_unstable();
                    semaphore_scratch.dedup();

                    match img.deref() {
                        ImageData::Imported(archandle) => unsafe {
                            archandle
                                .0
                                .get_header()
                                .object_data
                                .1
                                .get_mut(&image_storage_lock)
                                .synchronization
                                .update_state(
                                    usage.dst_queue_family,
                                    TypeSome::new_some(usage.dst_layout),
                                    &semaphore_scratch,
                                    *layout,
                                    *queue_family,
                                    img.is_sharing_concurrent(),
                                );
                        },
                        ImageData::Swapchain(archandle) => unsafe {
                            assert!(semaphore_scratch.len() == 1, "Swapchains use legacy semaphores and do not support multiple signals or waits, using a swapchain in multiple submissions is disallowed (you should really just transfer the final image into it at the end of the frame)");

                            let mut mutable = archandle
                                .0
                                .get_header()
                                .object_data
                                .get_mut(&swapchain_storage_lock);

                            let acquire_semaphore = self.swapchain_semaphores.next(&self.device);
                            let release_semaphore = self.swapchain_semaphores.next(&self.device);

                            let mut attempt = 0;
                            let index = loop {
                                let image_index = match mutable.acquire_image(
                                    u64::MAX,
                                    acquire_semaphore,
                                    vk::Fence::null(),
                                    &self.device,
                                ) {
                                    SwapchainAcquireStatus::Ok(index, false) => break index,
                                    SwapchainAcquireStatus::OutOfDate
                                    | SwapchainAcquireStatus::Ok(_, true) => {
                                        // we allow the swapchain to be recreated two times, then crash
                                        if attempt == 2 {
                                            panic!("Swapchain immediatelly invalid after being recreated two times");
                                        }

                                        if !waited_idle {
                                            self.device.wait_idle();
                                            waited_idle = true;
                                        }

                                        mutable
                                            .recreate(&archandle.0.get_header().info, &self.device)
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

                            let data = mutable.get_image_data(index);
                            submission_swapchains
                                .entry(semaphore_scratch[0])
                                .or_default()
                                .push(SwapchainPreset {
                                    swapchain: handle,
                                    vkhandle: mutable.get_swapchain(index),
                                    image: data.image,
                                    image_index: index,
                                    image_acquire: acquire_semaphore,
                                    image_release: release_semaphore,
                                });

                            // we do not update any synchronization state here, since we already get synchonization from swapchain image acquire and the subsequent present
                            swapchain_indices.insert(handle, index);
                        },
                        _ => unreachable!(),
                    }
                }
                _ => {}
            }
        }

        // FIXME code duplication
        for (i, buf) in self.buffers.iter().enumerate() {
            let handle = GraphBuffer::new(i);
            let data = self.get_buffer_meta(handle);

            if data.alive.get() == false {
                continue;
            }

            match buf.deref() {
                BufferData::Imported(archandle) => {
                    let resource = GraphResource::Buffer(handle);
                    let usage = accessors.get(&resource).unwrap();

                    let ResourceState::Normal { layout, queue_family, access } = &buffer_last_state[i].0 else {
                        panic!("Unsupported resource state (TODO should Uninit be allowed?)");
                    };

                    semaphore_scratch.clear();
                    semaphore_scratch.extend(usage.accessors.iter().map(|a| {
                        let pass = self.timeline[a.index()].get_pass().unwrap();
                        let submission =
                            self.get_pass_meta(pass).scheduled_submission.get().unwrap();
                        semaphores[submission as usize].0
                    }));

                    // nothing is using the resource, continue
                    // TODO is this even possible? wouldn't the resource be dead?
                    if semaphore_scratch.len() == 0 {
                        continue;
                    }

                    semaphore_scratch.sort_unstable();
                    semaphore_scratch.dedup();

                    unsafe {
                        archandle
                            .0
                            .get_header()
                            .object_data
                            .1
                            .get_mut(&buffer_storage_lock)
                            .synchronization
                            .update_state(
                                usage.dst_queue_family,
                                TypeNone::new_none(),
                                &semaphore_scratch,
                                *layout,
                                *queue_family,
                                buf.is_sharing_concurrent(),
                            );
                    }
                }
                _ => {}
            }
        }

        let mut raw_images = self
            .images
            .iter()
            .enumerate()
            .map(|(i, data)| unsafe {
                let image = GraphImage::new(i);

                if !self.get_image_meta(image).alive.get() {
                    return None;
                }

                match data.deref() {
                    ImageData::TransientPrototype(_, _) => None,
                    ImageData::Transient(physical) => {
                        Some(self.get_physical_image_data(*physical).vkhandle)
                    }
                    ImageData::Imported(archandle) => Some(archandle.0.get_handle()),
                    ImageData::Swapchain(archandle) => {
                        let image_index = *swapchain_indices.get(&image).unwrap();
                        Some(
                            archandle
                                .0
                                .get_header()
                                .object_data
                                .get(&swapchain_storage_lock)
                                .get_image_data(image_index)
                                .image,
                        )
                    }
                    ImageData::Moved(_, _, _) => todo!("TODO implement this"),
                }
            })
            .collect::<Vec<_>>();

        let mut raw_buffers = self
            .buffers
            .iter()
            .enumerate()
            .map(|(i, data)| unsafe {
                let buffer = GraphBuffer::new(i);

                if !self.get_buffer_meta(buffer).alive.get() {
                    return None;
                }

                match data.deref() {
                    BufferData::TransientPrototype(_, _) => None,
                    BufferData::Transient(physical) => {
                        Some(self.get_physical_buffer_data(*physical).vkhandle)
                    }
                    BufferData::Imported(archandle) => Some(archandle.0.get_handle()),
                }
            })
            .collect::<Vec<_>>();

        let current_generation = self.device.open_generation();
        current_generation.add_submissions(semaphores.iter().map(|(s, _)| *s));
        let id = current_generation.id();
        current_generation.finish();

        self.prev_generation.set(Some(id));

        // TODO multithreaded execution
        let mut memory_barriers = Vec::new();
        let mut image_barriers: Vec<vk::ImageMemoryBarrier2KHR> = Vec::new();
        let mut buffer_barriers = Vec::new();

        unsafe {
            for (i, sub) in submissions.into_iter().enumerate() {
                let info = vk::CommandBufferAllocateInfo {
                    command_pool: self.command_pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: 1,
                    ..Default::default()
                };
                let command_buffer = unsafe {
                    self.device
                        .device()
                        .allocate_command_buffers(&info)
                        .unwrap()[0]
                };

                let d = self.device.device();
                let Submission {
                    queue,
                    passes,
                    semaphore_dependencies,
                    barriers,
                    special_barriers,
                } = sub;

                d.begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        p_inheritance_info: std::ptr::null(),
                        ..Default::default()
                    },
                )
                .unwrap();

                let mut barriers = barriers.iter();
                let mut special_barriers = special_barriers.iter();

                for (i, pass) in passes.into_iter().enumerate() {
                    // extract into function things
                    do_barriers(
                        i,
                        d,
                        &mut barriers,
                        &mut memory_barriers,
                        &mut image_barriers,
                        &mut buffer_barriers,
                        &mut special_barriers,
                        &raw_images,
                        &raw_buffers,
                        command_buffer,
                        false,
                    );
                    let executor = GraphExecutor {
                        graph: self,
                        current_pass: pass,
                        command_buffer,
                        raw_images: &raw_images,
                        raw_buffers: &raw_buffers,
                    };
                    self.passes[pass.index()]
                        .pass
                        .borrow_mut()
                        .execute(&executor, &self.device);
                }
                do_barriers(
                    i,
                    d,
                    &mut barriers,
                    &mut memory_barriers,
                    &mut image_barriers,
                    &mut buffer_barriers,
                    &mut special_barriers,
                    &raw_images,
                    &raw_buffers,
                    command_buffer,
                    true,
                );

                d.end_command_buffer(command_buffer).unwrap();

                let pass_data = &self.passes[i];
                let queue = self.queues[pass_data.queue.index()].inner.clone();
                let (queue_submission, finished_semaphore) = semaphores[i];
                let swapchains = submission_swapchains.get(&queue_submission);

                let mut wait_semaphores = Vec::new();
                let mut signal_semaphores = Vec::new();
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
                signal_semaphores.push(vk::SemaphoreSubmitInfoKHR {
                    semaphore: finished_semaphore.raw,
                    value: finished_semaphore.value,
                    stage_mask: vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                    ..Default::default()
                });
                for dep in semaphore_dependencies {
                    let semaphore = semaphores[dep.index()].1;
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
                .map_err(|e| panic!("Submit err {:?}", e));

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

                    for res in results {
                        pumice::new_result((), res).unwrap();
                    }
                }
            }
        }

        // let mut file = OpenOptions::new()
        //     .write(true)
        //     .create(true)
        //     .truncate(true)
        //     .open("target/test.dot")
        //     .unwrap();
        // cargo test --quiet -- graph::test_graph --nocapture && cat target/test.dot | dot -Tpng -o target/out.png
        // self.write_dot_representation(&submissions, &mut file);
        // Self::write_submissions_dot_representation(&submissions, &mut file);
    }

    fn handle_resource<T: ResourceMarker>(
        &self,
        p: GraphPass,
        timeline_pass: TimelinePass,
        submission_pass: SubmissionPass,
        queue_family: u32,
        img: &T::Data,
        recorder: &RefCell<SubmissionRecorder>,
        image_rw: &mut Vec<(ResourceState<T>, bool)>,
        queued_ownership_transitions: &mut SmallVec<
            [(
                u32,
                Vec<(
                    GraphSubmission,
                    SubmissionPass,
                    SubmissionPass,
                    GraphResource,
                )>,
            ); 8],
        >,
        accessors: &mut ahash::HashMap<GraphResource, ResourceFirstAccess>,
    ) where
        // hack because without this the typechecker is not cooperating
        T::IfImage<vk::ImageLayout>: Copy,
    {
        let writes = self.emit_barriers::<T>(
            p,
            submission_pass,
            queue_family,
            img,
            recorder,
            image_rw,
            queued_ownership_transitions,
        );

        let resource = img.graph_resource();
        let imported = match resource {
            GraphResource::Image(handle) => {
                matches!(
                    self.get_image_data(handle),
                    ImageData::Imported(_) | ImageData::Swapchain(_)
                )
            }
            GraphResource::Buffer(handle) => {
                matches!(self.get_buffer_data(handle), BufferData::Imported(_))
            }
        };

        if imported {
            let ResourceFirstAccess {
                closed, accessors, ..
            } = accessors.entry(resource).or_insert_with(|| {
                let ResourceState::Normal {
                    layout,
                    queue_family,
                    access,
                } = &image_rw[img.raw_resource_handle().index()].0 else {
                    panic!("Impossible for external resources to be any other state")
                };

                ResourceFirstAccess {
                    closed: false,
                    accessors: smallvec![],
                    // TODO consider splitting images and buffers with ResourceMarker
                    dst_layout: layout.to_option().unwrap_or_default(),
                    dst_queue_family: *queue_family,
                }
            });

            if !*closed {
                if !writes || accessors.is_empty() {
                    if !accessors.contains(&timeline_pass) {
                        accessors.push(timeline_pass);
                    }
                }
                if writes {
                    *closed = true;
                }
            }
        }
    }

    fn get_last_resource_usage<'a, T: ResourceMarker>(
        &'a self,
        resource_last_state: &'a [(ResourceState<T>, bool)],
        scratch: &'a mut Vec<(GraphSubmission, SubmissionPass)>,
    ) -> impl Iterator<Item = ResourceLastUse> + 'a {
        resource_last_state.iter().map(|(state, _)| {
            fn add(
                from: impl Iterator<Item = PassTouch> + ExactSizeIterator + Clone,
                to: &mut Vec<(GraphSubmission, SubmissionPass)>,
                graph: &Graph,
            ) {
                let parts = from.map(|touch| {
                    let meta = graph.get_pass_meta(touch.pass);
                    let submission = GraphSubmission(meta.scheduled_submission.get().unwrap());
                    let pass = SubmissionPass(meta.scheduled_submission_position.get().unwrap());
                    (submission, pass)
                });
                to.extend(parts);
            };
            scratch.clear();
            match state {
                ResourceState::Uninit | ResourceState::Moved => {
                    return ResourceLastUse::None;
                }
                ResourceState::MoveDst { parts } => {
                    add(
                        parts.get().iter().map(|(touch, ..)| touch.clone()),
                        scratch,
                        self,
                    );
                }
                ResourceState::Normal {
                    layout,
                    queue_family,
                    access,
                } => add(access.iter().cloned(), scratch, self),
            }
            match scratch.len() {
                0 => ResourceLastUse::None,
                1 => ResourceLastUse::Single(scratch[0].0, smallvec![scratch[0].1]),
                _ => {
                    if scratch[1..].iter().all(|(sub, _)| *sub == scratch[0].0) {
                        ResourceLastUse::Single(
                            scratch[0].0,
                            scratch.iter().map(|(_, pass)| *pass).collect(),
                        )
                    } else {
                        use slice_group_by::GroupBy;
                        scratch.sort_unstable_by_key(|(sub, _)| *sub);
                        let groups = scratch.binary_group_by_key(|(sub, _)| *sub);
                        ResourceLastUse::Multiple(groups.map(|slice| slice[0].0).collect())
                    }
                }
            }
        })
    }

    fn emit_barriers<T: ResourceMarker>(
        &self,
        pass: GraphPass,
        dst_pass: SubmissionPass,
        dst_queue_family: u32,
        resource_data: &T::Data,
        recorder: &RefCell<SubmissionRecorder>,
        resource_rw: &mut [(ResourceState<T>, bool)],
        queued_ownership_transitions: &mut [(
            u32,
            Vec<(
                GraphSubmission,
                SubmissionPass,
                SubmissionPass,
                GraphResource,
            )>,
        )],
    ) -> bool
    where
        // hack because without this the typechecker is not cooperating
        T::IfImage<vk::ImageLayout>: Copy,
    {
        let raw_resource_handle = resource_data.raw_resource_handle();
        let resource_handle = resource_data.graph_resource();
        let &mut (ref mut state, concurrent) = &mut resource_rw[raw_resource_handle.index()];
        let dst_touch = PassTouch {
            pass,
            access: resource_data.access(),
            stages: resource_data.stages(),
        };
        let mut dst_writes = dst_touch.access.contains_write();
        let dst_layout = T::when_image(|| resource_data.start_layout());
        let normal_state = || ResourceState::Normal {
            layout: dst_layout,
            queue_family: dst_queue_family,
            access: smallvec![dst_touch],
        };
        match state {
            // no dependency
            ResourceState::Uninit => {
                *state = normal_state();
            }
            // if the current access only needs to read, add it to the readers (and handle layout transitions)
            // otherwise synchronize against all passes and transition to normal state
            ResourceState::MoveDst { parts } => {
                assert!(ImageMarker::IS_IMAGE, "Only images  can be moved");
                let dst_layout = dst_layout.unwrap();
                let parts = parts.get_mut();
                for (part, part_i) in parts.iter_mut().zip(0u32..) {
                    let (src_touch, src_layout, move_src) = (part.0, part.1.unwrap(), part.2);

                    let mut synchronized = false;
                    let mut synchronize = || {
                        if !synchronized {
                            let (touch, src_layout, move_src) = part;
                            // we need the previous access to finish before we can transition the layout
                            recorder.borrow_mut().add_dependency(
                                touch.pass,
                                touch.access,
                                touch.stages,
                                dst_touch.access,
                                dst_touch.stages,
                            );
                            synchronized = true;
                            touch.access = vk::AccessFlags2KHR::empty();
                        }
                    };

                    if dst_writes {
                        synchronize();
                        dst_writes = true;
                    }

                    if dst_layout != vk::ImageLayout::UNDEFINED && dst_layout != src_layout {
                        synchronize();
                        dst_writes = true;
                        recorder
                            .borrow_mut()
                            .layout_transition(move_src, src_layout..dst_layout);
                    }

                    // queue family ownership transition
                    let data = self.get_pass_data(src_touch.pass);
                    let src_queue_family = self.get_queue_family(data.queue);
                    // 1. concurrent images do not need ownership transition
                    // 2. if we do not need the contents of the resource (=layout UNDEFINED), we can simply ignore the transfer
                    //   https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-queue-transfers
                    // 3. no transfer is neccessary if both accesses are on the same queue
                    if !concurrent
                        /* && dst_layout != vk::ImageLayout::UNDEFINED */
                        && src_queue_family != dst_queue_family
                    {
                        synchronize();
                        dst_writes = true;
                        // ownership release and acquire is handled when the current submission is closed, push some data that
                        // we want to acquire ownership from src_queue_family

                        // find the vector of pending acquires for the src queue
                        let (_, entry) = queued_ownership_transitions
                            .iter_mut()
                            .find(|(f, _)| *f == src_queue_family)
                            .unwrap();

                        let src_meta = self.get_pass_meta(src_touch.pass);
                        let submission =
                            GraphSubmission(src_meta.scheduled_submission.get().unwrap());
                        let pass =
                            SubmissionPass(src_meta.scheduled_submission_position.get().unwrap());

                        entry.push((submission, pass, dst_pass, GraphResource::Image(move_src)));
                    }
                }

                let mut recorder = recorder.borrow_mut();

                // ???
                if dst_writes {
                    *state = normal_state();
                } else {
                    // TODO verify this, seems flaky
                    parts.retain(|p| !p.0.access.is_empty());
                    // using img.handle is dubious
                    parts.push((
                        dst_touch,
                        T::when_image(|| dst_layout),
                        GraphImage(raw_resource_handle.0),
                    ));
                }
            }
            &mut ResourceState::Normal {
                layout: ref mut src_layout,
                queue_family: ref mut src_queue_family,
                ref mut access,
            } => {
                let mut synchronized = false;
                let mut synchronize = || {
                    if !synchronized {
                        for touch in &*access {
                            // we need the previous access to finish before we can transition the layout
                            recorder.borrow_mut().add_dependency(
                                touch.pass,
                                touch.access,
                                touch.stages,
                                dst_touch.access,
                                dst_touch.stages,
                            );
                            synchronized = true;
                        }
                    }
                };

                if dst_writes {
                    synchronize();
                }

                // layout transition
                if ImageMarker::IS_IMAGE
                    && dst_layout.unwrap() != vk::ImageLayout::UNDEFINED
                    && dst_layout.unwrap() != src_layout.unwrap()
                {
                    synchronize();
                    dst_writes = true;
                    recorder.borrow_mut().layout_transition(
                        GraphImage(raw_resource_handle.0),
                        src_layout.unwrap()..dst_layout.unwrap(),
                    );
                    *src_layout = dst_layout;
                }

                // queue family ownership transition
                if !concurrent && *src_queue_family != dst_queue_family {
                    synchronize();
                    dst_writes = true;

                    // ownership release and acquire is handled when the current submission is closed, push some data that
                    // we want to acquire ownership from src_queue_family

                    // find the vector of pending acquires for the src queue
                    let (_, entry) = queued_ownership_transitions
                        .iter_mut()
                        .find(|(f, _)| *f == *src_queue_family)
                        .unwrap();

                    for src_touch in &*access {
                        let src_meta = self.get_pass_meta(src_touch.pass);
                        let submission =
                            GraphSubmission(src_meta.scheduled_submission.get().unwrap());
                        let pass =
                            SubmissionPass(src_meta.scheduled_submission_position.get().unwrap());

                        entry.push((submission, pass, dst_pass, resource_handle));

                        *src_queue_family = dst_queue_family;
                    }
                }

                if dst_writes {
                    // already synchronized
                    access.clear();
                }
                access.push(dst_touch);
            }
            // TODO perhaps this shouldn't be a hard error and instead delegate access to the move destination
            ResourceState::Moved => panic!("Attempt to access moved resource"),
        }
        dst_writes
    }
    fn write_dot_representation(
        &mut self,
        submissions: &Vec<Submission>,
        writer: &mut dyn std::io::Write,
    ) {
        let clusters = Fun::new(|w| {
            writeln!(w);

            // q0[label="Queue 0:"; peripheries=0; fontsize=15; fontname="Helvetica,Arial,sans-serif bold"];
            for q in 0..self.queues.len() {
                write!(
                    w,
                    r#"q{q}[label="{}:"; peripheries=0; fontsize=15; fontname="Helvetica,Arial,sans-serif bold"];"#,
                    self.get_queue_display(GraphQueue::new(q))
                        .set_prefix("Queue ")
                );
            }

            writeln!(w);

            let queue_submitions = |queue_index: usize| {
                submissions
                    .iter()
                    .filter(move |sub| sub.queue.index() == queue_index)
                    .flat_map(|sub| &sub.passes)
                    .cloned()
            };

            // subgraph cluster_0 {
            //     style=dashed;
            //     p0[label="#0"];
            //     p1[label="#1"];
            // }
            for (i, sub) in submissions.iter().enumerate() {
                let nodes = Fun::new(|w| {
                    for &p in &sub.passes {
                        write!(
                            w,
                            r#"p{}[label="{}"];"#,
                            p.index(),
                            self.get_pass_display(p).set_prefix("#")
                        );
                    }
                    Ok(())
                });

                token_abuse!(
                    w,
                    "subgraph cluster_$ {\
                        style=dashed;\
                        $\
                    }",
                    i,
                    nodes
                );
            }

            writeln!(w);

            // next edges will serve as layout constraints, don't show them
            write!(w, "edge[style=invis];");

            writeln!(w);

            let heads = (0..self.queues.len())
                .map(|q| queue_submitions(q).next())
                .collect::<Vec<_>>();

            // make sure that queues start vertically aligned
            if self.queues.len() > 1 {
                for q in 1..self.queues.len() {
                    write!(w, "q{} -> q{}[weight=99];", q - 1, q);
                }
            }

            // a weird way to count that heads has at least two Some(_)
            if heads.iter().flatten().nth(1).is_some() {
                let mut heads = heads
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| p.as_ref().map(|p| (i, p)));

                for (q, p) in heads {
                    write!(w, "q{} -> p{};", q, p.index());
                }
            }

            writeln!(w);

            // p0 -> p1 -> p2;
            for q in 0..self.queues.len() {
                if queue_submitions(q).nth(1).is_some() {
                    let mut first = true;
                    for p in queue_submitions(q) {
                        if !first {
                            write!(w, " -> ");
                        }
                        write!(w, "p{}", p.index());
                        first = false;
                    }
                    write!(w, ";");
                }
            }

            writeln!(w);

            // { rank = same; q1; p2; p3; }
            for q in 0..self.queues.len() {
                let nodes = Fun::new(|w| {
                    write!(w, "q{q}; ");

                    for p in queue_submitions(q) {
                        write!(w, "p{}; ", p.index());
                    }
                    Ok(())
                });

                token_abuse!(w, (nodes)
                    {
                        rank = same;
                        $
                    }
                );
            }

            writeln!(w);
            write!(w, "edge[style=filled];");
            writeln!(w);

            // the visible edges for the actual dependencies
            for q in 0..self.queues.len() {
                for p in queue_submitions(q) {
                    let dependencies = &self.get_pass_data(p).dependencies;
                    for dep in dependencies {
                        write!(w, "p{} -> p{}", dep.index(), p.index());
                        if !dep.is_hard() {
                            write!(w, "[color=darkgray]");
                        }
                        write!(w, ";");
                    }
                }
            }

            Ok(())
        });
        let mut writer = WeirdFormatter::new(writer);
        token_abuse!(
            writer,
            (clusters)
            digraph G {
                fontname="Helvetica,Arial,sans-serif";
                node[fontname="Helvetica,Arial,sans-serif"; fontsize=9; shape=rect];
                edge[fontname="Helvetica,Arial,sans-serif"];

                newrank = true;
                rankdir = TB;

                $
            }
        );
    }
    fn write_submissions_dot_representation(
        submissions: &Vec<Submission>,
        writer: &mut dyn std::io::Write,
    ) {
        let clusters = Fun::new(|w| {
            for (i, sub) in submissions.iter().enumerate() {
                write!(w, r##"s{i}[label="{i} (q{})"];"##, sub.queue.index());
                if !sub.semaphore_dependencies.is_empty() {
                    write!(w, "{{");
                    for p in &sub.semaphore_dependencies {
                        write!(w, "s{} ", p.index());
                    }
                    write!(w, "}} -> s{i};");
                }
            }
            Ok(())
        });
        let mut writer = WeirdFormatter::new(writer);
        token_abuse!(
            writer,
            (clusters)
            digraph G {
                fontname="Helvetica,Arial,sans-serif";
                node[fontname="Helvetica,Arial,sans-serif"; fontsize=9; shape=rect];
                edge[fontname="Helvetica,Arial,sans-serif"];

                newrank = true;
                rankdir = TB;

                $
            }
        );
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let Self {
            memory,
            prev_generation,
            swapchain_semaphores,
            command_pool,
            device,
            ..
        } = self;

        let memory = unsafe { ManuallyDrop::take(&mut *memory.borrow_mut()) };
        let command_pool = *command_pool;

        let device_ptr: *const Device = Arc::as_ptr(&device.0);
        let owned_device = unsafe { ManuallyDrop::take(device) };

        if let Some(prev) = prev_generation.get() {
            unsafe {
                (*device_ptr).add_inflight(
                    prev,
                    std::iter::once(InflightResource::Closure(ManuallyDrop::new(Box::new(
                        move || {
                            owned_device.device().destroy_command_pool(
                                command_pool,
                                owned_device.allocator_callbacks(),
                            );
                            for mem in *memory {
                                for RcPtrComparator(block) in mem.allocator.collect_blocks() {
                                    owned_device.allocator().free_memory(block.allocation);
                                }
                            }
                        },
                    )))),
                )
            }
        }
    }
}

unsafe fn do_barriers(
    i: usize,
    d: &DeviceWrapper,
    barriers: &mut std::slice::Iter<(SubmissionPass, SimpleBarrier)>,
    memory_barriers: &mut Vec<vk::MemoryBarrier2KHR>,
    image_barriers: &mut Vec<vk::ImageMemoryBarrier2KHR>,
    buffer_barriers: &mut Vec<vk::BufferMemoryBarrier2KHR>,
    special_barriers: &mut std::slice::Iter<(SubmissionPass, SpecialBarrier)>,
    raw_images: &[Option<vk::Image>],
    raw_buffers: &[Option<vk::Buffer>],
    command_buffer: vk::CommandBuffer,
    flush: bool,
) {
    memory_barriers.clear();
    while let Some((pass, barrier)) = barriers.clone().next() {
        if flush || pass.index() == i {
            barriers.next();
            memory_barriers.push(vk::MemoryBarrier2KHR {
                src_stage_mask: barrier.src_stages,
                src_access_mask: barrier.src_access,
                dst_stage_mask: barrier.dst_stages,
                dst_access_mask: barrier.dst_access,
                ..Default::default()
            });
        } else {
            break;
        }
    }
    image_barriers.clear();
    buffer_barriers.clear();
    while let Some((pass, barrier)) = special_barriers.clone().next() {
        if flush || pass.index() == i {
            special_barriers.next();
            // TODO support moves - ie non-whole subresource ranges
            let subresource_range = vk::ImageSubresourceRange {
                // TODO don't be dumb
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            };
            match barrier {
                SpecialBarrier::LayoutTransition {
                    image,
                    src_layout,
                    dst_layout,
                } => image_barriers.push(vk::ImageMemoryBarrier2KHR {
                    old_layout: *src_layout,
                    new_layout: *dst_layout,
                    image: raw_images[image.index()].unwrap(),
                    subresource_range,
                    ..Default::default()
                }),
                SpecialBarrier::ImageOwnershipTransition {
                    image,
                    src_family,
                    dst_family,
                } => image_barriers.push(vk::ImageMemoryBarrier2KHR {
                    src_queue_family_index: *src_family,
                    dst_queue_family_index: *dst_family,
                    image: raw_images[image.index()].unwrap(),
                    subresource_range,
                    ..Default::default()
                }),
                SpecialBarrier::BufferOwnershipTransition {
                    buffer,
                    src_family,
                    dst_family,
                } => buffer_barriers.push(vk::BufferMemoryBarrier2KHR {
                    src_queue_family_index: *src_family,
                    dst_queue_family_index: *dst_family,
                    buffer: raw_buffers[buffer.index()].unwrap(),
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                    ..Default::default()
                }),
            };

            // if let Some(last) = image_barriers.last_mut() {
            //     if last.image == barrier.image {
            //         // TODO merge barriers
            //         continue;
            //     }
            // }
        } else {
            break;
        }
    }
    if !(memory_barriers.is_empty() && image_barriers.is_empty() && buffer_barriers.is_empty()) {
        d.cmd_pipeline_barrier_2_khr(
            command_buffer,
            &vk::DependencyInfoKHR {
                // TODO track opportunities to use BY_REGION and friends
                // https://stackoverflow.com/questions/65471677/the-meaning-and-implications-of-vk-dependency-by-region-bit
                dependency_flags: vk::DependencyFlags::empty(),
                memory_barrier_count: memory_barriers.len() as u32,
                p_memory_barriers: memory_barriers.as_ptr(),
                buffer_memory_barrier_count: buffer_barriers.len() as u32,
                p_buffer_memory_barriers: buffer_barriers.as_ptr(),
                image_memory_barrier_count: image_barriers.len() as u32,
                p_image_memory_barriers: image_barriers.as_ptr(),
                ..Default::default()
            },
        );
    }
}

fn submission_fill_reuse<T: ResourceMarker>(
    resource_last_touch: impl Iterator<Item = ResourceLastUse>,
    resource_availability: &mut [AvailabilityToken],
    submission_reuse: &mut [SubmissionResourceReuse],
    // <the asociated submission, (number of parents currently counted by dfs, latest parent submission so that it isn't countet twice)>
    intersection_hashmap: &mut ahash::HashMap<GraphSubmission, (GraphSubmission, u32)>,
    submission_children: &ImmutableGraph,
    (counter, interner): &mut (
        AvailabilityToken,
        ahash::HashMap<ResourceLastUse, AvailabilityToken>,
    ),
) {
    // we are (ab)using ResourceMarker to be generic over images
    // using branches on T::IS_IMAGE which will get constant folded
    for (i, reuse) in resource_last_touch.enumerate() {
        if reuse == ResourceLastUse::None {
            continue;
        }
        // TODO remove this clone when raw_entry_mut is stabilized (if ever ;_; )
        let token = *interner.entry(reuse.clone()).or_insert_with(|| {
            let old = *counter;
            counter.bump();
            old
        });
        resource_availability[i] = token;
        let handle = RawHandle::new(i);
        match reuse {
            ResourceLastUse::None => {}
            ResourceLastUse::Single(sub, passes) => {
                let sub = &mut submission_reuse[sub.index()];
                sub.after_local_passes.push((token, passes));
            }
            ResourceLastUse::Multiple(submissions) => {
                // we need to find nodes that depend on all of these submissions
                // the graph being transitively reduced helps us do less work
                intersection_hashmap.clear();
                let len = submissions.len() as u32;
                for &sub in &submissions {
                    submission_children.dfs_visit(sub.0, |node| {
                        let (current_parent, entry) = intersection_hashmap
                            .entry(GraphSubmission(node))
                            .or_insert((sub, 0));

                        // we've already visited it, entry 0 if this is the first time we've encountered it
                        if *current_parent == sub && *entry != 0 {
                            return DFSCommand::Ascend;
                        }

                        *entry += 1;
                        *current_parent = sub;
                        if *entry == len {
                            return DFSCommand::Ascend;
                        }

                        DFSCommand::Continue
                    });
                }
                for (&sub, &(_, count)) in intersection_hashmap.iter() {
                    // every submission inserted the node as its child, this means that all of the submissions share it
                    if count == len {
                        let sub = &mut submission_reuse[sub.index()];
                        sub.immediate.push(token);
                    }
                }
            }
        }
    }
}

#[test]
fn test_graph() {
    let device = unsafe { __test_init_device(true) };
    let mut g = Graph::new(device);
    g.run(|b| {
        let dummy_queue1 = submission::Queue::new(pumice::vk10::Queue::from_raw(1), 0);
        let dummy_queue2 = submission::Queue::new(pumice::vk10::Queue::from_raw(2), 0);
        let dummy_queue3 = submission::Queue::new(pumice::vk10::Queue::from_raw(3), 0);

        let q0 = b.import_queue(dummy_queue1);
        let q1 = b.import_queue(dummy_queue2);
        let q2 = b.import_queue(dummy_queue3);

        let p0 = b.add_pass(q0, |_: &mut GraphPassBuilder, _: &Device| -> () {}, "p0");
        let p1 = b.add_pass(q0, |_: &mut GraphPassBuilder, _: &Device| {}, "p1");

        let p2 = b.add_pass(q1, |_: &mut GraphPassBuilder, _: &Device| {}, "p2");
        let p3 = b.add_pass(q1, |_: &mut GraphPassBuilder, _: &Device| {}, "p3");

        let p4 = b.add_pass(q2, |_: &mut GraphPassBuilder, _: &Device| {}, "p4");

        b.add_pass_dependency(p0, p1, true, true);
        b.add_pass_dependency(p0, p2, true, true);
        b.add_pass_dependency(p2, p3, true, true);

        b.add_pass_dependency(p0, p4, true, true);
        b.add_pass_dependency(p3, p4, true, true);

        b.force_pass_run(p1);
        b.force_pass_run(p2);
        b.force_pass_run(p3);
        b.force_pass_run(p4);
    });
}

#[derive(Clone)]
pub struct PassImageData {
    handle: GraphImage,
    usage: vk::ImageUsageFlags,
    stages: vk::PipelineStageFlags2KHR,
    access: vk::AccessFlags2KHR,
    start_layout: vk::ImageLayout,
    end_layout: Option<vk::ImageLayout>,
}

impl PassImageData {
    fn is_written(&self) -> bool {
        self.access.contains_write()
    }
}

pub struct PassBufferData {
    handle: GraphBuffer,
    usage: vk::BufferUsageFlags,
    access: vk::AccessFlags2KHR,
    stages: vk::PipelineStageFlags2KHR,
}

impl PassBufferData {
    fn is_written(&self) -> bool {
        self.access.contains_write()
    }
}

struct PassData {
    queue: GraphQueue,
    force_run: bool,
    images: Vec<PassImageData>,
    buffers: Vec<PassBufferData>,
    stages: vk::PipelineStageFlags2KHR,
    access: vk::AccessFlags2KHR,
    dependencies: Vec<PassDependency>,
    pass: RefCell<Box<dyn ObjectSafePass>>,
}

impl PassData {
    fn add_dependency(&mut self, dependency: GraphPass, hard: bool, real: bool) {
        if let Some(found) = self
            .dependencies
            .iter_mut()
            .find(|d| d.get_pass() == dependency)
        {
            // hard dependency overwrites a soft one
            if hard {
                found.set_hard(true);
            }
            if real {
                found.set_real(true);
            }
        } else {
            self.dependencies
                .push(PassDependency::new(dependency, hard, real));
        }
    }
}

struct ImageMove {
    // we currently only allow moving images of same format and extent
    // and then concatenate their layers in the input order
    from: SmallVec<[GraphImage; 4]>,
    to: GraphImage,
}

// TODO merge ImageData and BufferData with ResourceMarker
#[derive(Clone)]
enum ImageData {
    // FIXME ImageCreateInfo is large and this scheme is weird
    TransientPrototype(object::ImageCreateInfo, AllocationCreateInfo),
    Transient(PhysicalImage),
    Imported(object::Image),
    Swapchain(object::Swapchain),
    // dst image, layer offset, layer count
    // TODO make GraphImage a Cell and update it when traversing to be a straight reference to the dst image
    // (if the dst image itself has  been moved, we will have to visit its dst image to get the final layer offset and such)
    Moved(GraphImage, u16, u16),
}
impl ImageData {
    fn get_variant_name(&self) -> &'static str {
        match self {
            ImageData::TransientPrototype(..) => "TransientPrototype",
            ImageData::Transient(..) => "Transient",
            ImageData::Imported(_) => "Imported",
            ImageData::Swapchain(_) => "Swapchain",
            ImageData::Moved { .. } => "Moved",
        }
    }
    fn is_sharing_concurrent(&self) -> bool {
        match self {
            ImageData::TransientPrototype(info, _) => info.sharing_mode_concurrent,
            ImageData::Imported(handle) => {
                unsafe { handle.0.get_header() }
                    .info
                    .sharing_mode_concurrent
            }
            ImageData::Swapchain(_) => false,
            ImageData::Transient(_) => unreachable!(),
            ImageData::Moved(..) => unreachable!(),
        }
    }
}
enum BufferData {
    TransientPrototype(object::BufferCreateInfo, AllocationCreateInfo),
    Transient(PhysicalBuffer),
    Imported(object::Buffer),
}

impl BufferData {
    fn is_sharing_concurrent(&self) -> bool {
        match self {
            BufferData::TransientPrototype(info, _) => info.sharing_mode_concurrent,
            BufferData::Transient(_) => unreachable!(),
            BufferData::Imported(handle) => {
                unsafe { handle.0.get_header() }
                    .info
                    .sharing_mode_concurrent
            }
        }
    }
}

struct GraphObject<T> {
    name: Option<Cow<'static, str>>,
    inner: T,
}

impl<T> GraphObject<T> {
    fn get_inner(&self) -> &T {
        &self.inner
    }
    fn get_inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }
    fn display(&self, index: usize) -> GraphObjectDisplay<'_> {
        GraphObjectDisplay {
            name: &self.name,
            index,
            prefix: "#",
        }
    }
}

impl<T> Deref for GraphObject<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for GraphObject<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

struct GraphObjectDisplay<'a> {
    name: &'a Option<Cow<'static, str>>,
    index: usize,
    prefix: &'a str,
}

impl<'a> GraphObjectDisplay<'a> {
    fn set_prefix(mut self, prefix: &'a str) -> Self {
        self.prefix = prefix;
        self
    }
}

impl<'a> Display for GraphObjectDisplay<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = self.name {
            write!(f, "{}", name.as_ref())
        } else {
            write!(f, "{}{}", self.prefix, self.index)
        }
    }
}

trait IntoObject<T>: Named<T> {
    fn into_object(self) -> GraphObject<T> {
        let (inner, name) = self.decompose();
        GraphObject { name, inner }
    }
    fn map_to_object<P, F: FnOnce(T) -> P>(self, fun: F) -> GraphObject<P> {
        let (inner, name) = self.map_named(fun).decompose();
        GraphObject { name, inner }
    }
}
impl<T, P: Named<T>> IntoObject<T> for P {}

pub trait Named<T>: Sized {
    type Map<P>: Named<P>;
    fn map_named<P, F: FnOnce(T) -> P>(self, fun: F) -> Self::Map<P>;
    fn decompose(self) -> (T, Option<Cow<'static, str>>);
}
impl<T> Named<T> for T {
    type Map<A> = A;
    fn map_named<P, F: FnOnce(T) -> P>(self, fun: F) -> Self::Map<P> {
        fun(self)
    }
    fn decompose(self) -> (T, Option<Cow<'static, str>>) {
        (self, None)
    }
}
impl<T> Named<T> for (T, &'static str) {
    type Map<A> = (A, &'static str);
    fn map_named<P, F: FnOnce(T) -> P>(self, fun: F) -> Self::Map<P> {
        (fun(self.0), self.1)
    }
    fn decompose(self) -> (T, Option<Cow<'static, str>>) {
        (self.0, Some(Cow::Borrowed(self.1)))
    }
}
impl<T> Named<T> for (T, String) {
    type Map<A> = (A, String);
    fn map_named<P, F: FnOnce(T) -> P>(self, fun: F) -> Self::Map<P> {
        (fun(self.0), self.1)
    }
    fn decompose(self) -> (T, Option<Cow<'static, str>>) {
        (self.0, Some(Cow::Owned(self.1)))
    }
}

#[repr(transparent)]
pub struct GraphBuilder(Graph);
impl GraphBuilder {
    pub fn acquire_swapchain(&mut self, swapchain: impl Named<object::Swapchain>) -> GraphImage {
        let handle = GraphImage::new(self.0.images.len());
        self.0.image_meta.push(ImageMeta::new());
        self.0
            .images
            .push(swapchain.map_to_object(ImageData::Swapchain));
        handle
    }
    pub fn import_queue(&mut self, queue: impl Named<submission::Queue>) -> GraphQueue {
        let object = queue.into_object();
        if let Some(i) = self
            .0
            .queues
            .iter()
            .position(|q| q.inner.raw() == object.inner.raw())
        {
            GraphQueue::new(i)
        } else {
            let handle = GraphQueue::new(self.0.queues.len());
            self.0.queues.push(object);
            handle
        }
    }
    pub fn import_image(&mut self, image: impl Named<object::Image>) -> GraphImage {
        let handle = GraphImage::new(self.0.images.len());
        self.0.image_meta.push(ImageMeta::new());
        self.0.images.push(image.map_to_object(ImageData::Imported));
        handle
    }
    pub fn import_buffer(&mut self, buffer: impl Named<object::Buffer>) -> GraphBuffer {
        let handle = GraphBuffer::new(self.0.buffers.len());
        self.0.buffer_meta.push(BufferMeta::new());
        self.0
            .buffers
            .push(buffer.map_to_object(BufferData::Imported));
        handle
    }
    pub fn create_image(
        &mut self,
        info: impl Named<object::ImageCreateInfo>,
        allocation: pumice_vma::AllocationCreateInfo,
    ) -> GraphImage {
        let handle = GraphImage::new(self.0.images.len());
        self.0.image_meta.push(ImageMeta::new());
        self.0
            .images
            .push(info.map_to_object(|a| ImageData::TransientPrototype(a, allocation)));
        handle
    }
    pub fn create_buffer(
        &mut self,
        info: impl Named<object::BufferCreateInfo>,
        allocation: pumice_vma::AllocationCreateInfo,
    ) -> GraphBuffer {
        let handle = GraphBuffer::new(self.0.buffers.len());
        self.0.buffer_meta.push(BufferMeta::new());
        self.0
            .buffers
            .push(info.map_to_object(|a| BufferData::TransientPrototype(a, allocation)));
        handle
    }
    #[track_caller]
    pub fn move_image<T: IntoIterator<Item = GraphImage>>(
        &mut self,
        images: impl Named<T>,
    ) -> GraphImage {
        let image = GraphImage::new(self.0.images.len());

        let object = images.map_to_object(|a| {
            let images = a.into_iter().collect::<SmallVec<_>>();

            // TODO perhaps it would be useful to move into already instantiated non-transient images
            let invalid_data_panic = |image: GraphImage, data: &ImageData| {
                panic!(
                    "Only Transient images can be moved, image '{}' has state '{}'",
                    self.0.get_image_display(image),
                    data.get_variant_name()
                )
            };

            let ImageData::TransientPrototype(mut first_info, mut first_allocation) = self.0.get_image_data(images[0]).clone() else {
                invalid_data_panic(images[0], self.0.get_image_data(images[0]))
            };

            // check that all of them are transient and that they have the same format and extent
            for &i in &images[1..] {
                let data = &self.0.get_image_data(i);
                let ImageData::TransientPrototype(info, allocation) = data else {
                    invalid_data_panic(i, data)
                };

                first_info.usage |= info.usage;

                assert_eq!(first_info.size, info.size);
                assert_eq!(first_info.format, info.format);
                assert_eq!(first_info.samples, info.samples);
                assert_eq!(first_info.tiling, info.tiling);
                assert_eq!(
                    first_info.sharing_mode_concurrent,
                    info.sharing_mode_concurrent
                );

                first_allocation.flags |= allocation.flags;
                first_allocation.required_flags |= allocation.required_flags;
                first_allocation.preferred_flags |= allocation.preferred_flags;
                first_allocation.memory_type_bits &= allocation.memory_type_bits;
                // FIXME user data is ignored
                first_allocation.priority = first_allocation.priority.max(allocation.priority);

                assert_eq!(first_allocation.pool, allocation.pool);
                // TODO try to unify usages
                assert_eq!(first_allocation.usage, allocation.usage);
            }

            // update the states of the move sources
            let mut layer_offset = 0;
            for &i in &images {
                let ImageData::TransientPrototype(info, _) = self.0.get_image_data(i) else {
                    unreachable!()
                };

                let layer_count: u16 = info.array_layers.try_into().unwrap();
                *self.0.images[i.index()].get_inner_mut() =
                    ImageData::Moved(image, layer_offset, layer_count);
                layer_offset += layer_count;
            }

            let mut info = ImageCreateInfo {
                array_layers: layer_offset as u32,
                ..first_info
            };

            self.0
                .timeline
                .push(GraphPassEvent::new(PassEventData::Move(
                    GraphPassMove::new(self.0.moves.len()),
                )));
            self.0.moves.push(ImageMove {
                from: images,
                to: image,
            });

            ImageData::TransientPrototype(info, first_allocation)
        });

        self.0.images.push(object);

        image
    }
    pub fn add_pass<T: CreatePass, N: Into<Cow<'static, str>>>(
        &mut self,
        queue: GraphQueue,
        pass: T,
        name: N,
    ) -> GraphPass {
        // we don't use impl Named here because it breaks type inference and we cannot name the closure type to resolve it so it makes the trait unusable
        let handle = GraphPass::new(self.0.passes.len());
        let data = {
            let mut builder = GraphPassBuilder::new(self, handle);
            let pass = pass.create(&mut builder, &self.0.device);
            builder.finish(queue, pass)
        };
        // if the string is 0, we set the name to None
        let name = Some(name.into()).filter(|n| !n.is_empty());
        self.0.passes.push(GraphObject { name, inner: data });
        self.0
            .timeline
            .push(GraphPassEvent::new(PassEventData::Pass(handle)));
        handle
    }
    pub fn add_pass_dependency(
        &mut self,
        first: GraphPass,
        then: GraphPass,
        hard: bool,
        real: bool,
    ) {
        self.0.passes[then.index()].add_dependency(first, hard, real);
    }
    pub fn force_pass_run(&mut self, pass: GraphPass) {
        self.0.passes[pass.index()].force_run = true;
    }
    pub fn add_scheduling_barrier(&mut self, queue: GraphQueue) {
        self.0
            .timeline
            .push(GraphPassEvent::new(PassEventData::Flush(queue)));
    }
}

pub struct GraphPassBuilder<'a> {
    graph_builder: &'a GraphBuilder,
    pass: GraphPass,
    images: Vec<PassImageData>,
    buffers: Vec<PassBufferData>,
    dependencies: Vec<PassDependency>,
}

impl<'a> GraphPassBuilder<'a> {
    fn new(graph_builder: &'a GraphBuilder, pass: GraphPass) -> Self {
        Self {
            graph_builder,
            pass,
            images: Vec::new(),
            buffers: Vec::new(),
            dependencies: Vec::new(),
        }
    }
    fn finish<T: RenderPass>(self, queue: GraphQueue, pass: T) -> PassData {
        let mut access = vk::AccessFlags2KHR::default();
        let mut stages = vk::PipelineStageFlags2KHR::default();
        for i in &self.images {
            if i.access.contains_write() {
                access |= i.access;
                stages |= i.stages;
            }
        }
        for b in &self.buffers {
            if b.access.contains_write() {
                access |= b.access;
                stages |= b.stages;
            }
        }

        PassData {
            queue: queue,
            force_run: false,
            images: self.images,
            buffers: self.buffers,
            stages,
            access,
            dependencies: self.dependencies,
            pass: RefCell::new(Box::new(StoredPass(Some(pass)))),
        }
    }
    // TODO check (possibly update for transients) usage flags against their create info
    pub fn use_image(
        &mut self,
        image: GraphImage,
        usage: vk::ImageUsageFlags,
        stages: vk::PipelineStageFlags2KHR,
        access: vk::AccessFlags2KHR,
        start_layout: vk::ImageLayout,
        end_layout: Option<vk::ImageLayout>,
    ) {
        // TODO deduplicate or explicitly forbid multiple entries with the same handle
        self.images.push(PassImageData {
            handle: image,
            usage,
            stages,
            access,
            start_layout,
            end_layout,
        });
    }
    pub fn use_buffer(
        &mut self,
        buffer: GraphBuffer,
        usage: vk::BufferUsageFlags,
        stages: vk::PipelineStageFlags2KHR,
        access: vk::AccessFlags2KHR,
    ) {
        // TODO deduplicate or explicitly forbid multiple entries with the same handle
        self.buffers.push(PassBufferData {
            handle: buffer,
            usage,
            access,
            stages,
        });
    }
}

pub struct GraphExecutor<'a> {
    graph: &'a Graph,
    current_pass: GraphPass,
    command_buffer: vk::CommandBuffer,
    raw_images: &'a [Option<vk::Image>],
    raw_buffers: &'a [Option<vk::Buffer>],
}

pub struct DescriptorState;
impl<'a> GraphExecutor<'a> {
    pub fn get_image_subresource_range(&self, handle: GraphImage) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            // TODO don't be dumb
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }
    pub fn get_image(&self, image: GraphImage) -> vk::Image {
        self.raw_images[image.index()].unwrap()
    }
    pub fn get_buffer(&self, buffer: GraphBuffer) -> vk::Buffer {
        self.raw_buffers[buffer.index()].unwrap()
    }
    pub fn make_descriptor(&self) -> DescriptorState {
        todo!()
    }
    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }
}

#[macro_export]
macro_rules! simple_handle {
    ($($visibility:vis $name:ident),+) => {
        $(
            #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
            #[repr(transparent)]
            $visibility struct $name(u32);
            impl $name {
                fn new(index: usize) -> Self {
                    assert!(index <= u32::MAX as usize);
                    Self(index as u32)
                }
                #[inline]
                fn index(&self) -> usize {
                    self.0 as usize
                }
                #[inline]
                fn to_raw(&self) -> $crate::graph::RawHandle {
                    $crate::graph::RawHandle(self.0)
                }
                #[inline]
                fn from_raw(raw: $crate::graph::RawHandle) -> Self {
                    Self(raw.0)
                }
            }
        )+
    };
}

pub use simple_handle;

simple_handle! {
    pub RawHandle,
    pub GraphQueue, pub GraphPass, pub GraphImage, pub GraphBuffer,
    GraphSubmission, PhysicalImage, PhysicalBuffer, GraphPassMove,
    // like a GraphPassEvent but only ever points to a pass
    TimelinePass,
    // the pass in a submission
    SubmissionPass,
    // vma allocations
    GraphAllocation,
    GraphMemoryTypeHandle
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct MemoryTypeIndex(u32);

macro_rules! optional_index {
    ($($name:ident),+) => {
        $(
            #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
            struct $name(OptionalU32);
            impl $name {
                const NONE: Self = Self(OptionalU32::NONE);
                fn new(index: Option<usize>) -> Self {
                    Self(OptionalU32::new(index.map(|i| {
                        assert!(i <= u32::MAX as usize); i as u32
                    })))
                }
                fn index(&self) -> Option<usize> {
                    self.0.get().map(|i| i as usize)
                }
            }
        )+
    };
}

optional_index! { QueueIntervals, GraphPassOption }

#[derive(Clone, Copy, PartialEq, Eq)]
struct GraphPassEventConfig;
impl Config for GraphPassEventConfig {
    const FIRST_BITS: usize = 2;
    const SECOND_BITS: usize = 30;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PassEventData {
    Pass(GraphPass),
    Move(GraphPassMove),
    Flush(GraphQueue),
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct GraphPassEvent(PackedUint<GraphPassEventConfig, u32>);

impl GraphPassEvent {
    fn to_timeline_pass(self) -> Option<TimelinePass> {
        self.get_pass().map(|p| TimelinePass(p.0))
    }
}

macro_rules! gen_pass_getters {
    ($name:ident, $data:ident: $($fun:ident, $handle:ident, $disc:tt: $val:expr;)+) => {
        impl $name {
            fn new(data: $data) -> Self {
                let (discriminant, index) = match data {
                    $(
                        $data::$disc(handle) => ($val, handle.0),
                    )+
                };
                Self(PackedUint::new(
                    discriminant,
                    index
                ))
            }
            $(
                fn $fun (&self) -> Option<$handle> {
                    if self.0.first() == $val {
                        Some($handle(self.0.second()))
                    } else {
                        None
                    }
                }
            )+
            fn get(&self) -> $data {
                let second = self.0.second();
                match self.0.first() {
                    $(
                        $val => $data::$disc($handle(second)),
                    )+
                    _ => unreachable!("gen_pass_getters! macro is missing enum variants")
                }
            }
        }
    };
}

gen_pass_getters! {
    GraphPassEvent, PassEventData:
    get_pass, GraphPass, Pass: 0;
    get_move, GraphPassMove, Move: 1;
    get_flush, GraphQueue, Flush: 2;
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct CombinedResourceConfig;
impl Config for CombinedResourceConfig {
    const FIRST_BITS: usize = 1;
    const SECOND_BITS: usize = 31;
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct CombinedResourceHandle(PackedUint<CombinedResourceConfig, u32>);

impl CombinedResourceHandle {
    fn new_image(image: GraphImage) -> Self {
        Self(PackedUint::new(0, image.0))
    }
    fn new_buffer(buffer: GraphBuffer) -> Self {
        Self(PackedUint::new(1, buffer.0))
    }
    fn get_image(&self) -> Option<GraphPass> {
        if self.0.first() == 0 {
            Some(GraphPass(self.0.second()))
        } else {
            None
        }
    }
    fn get_buffer(&self) -> Option<GraphPass> {
        if self.0.first() == 1 {
            Some(GraphPass(self.0.second()))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PassDependencyConfig;
impl Config for PassDependencyConfig {
    const FIRST_BITS: usize = 30;
    const SECOND_BITS: usize = 2;
}

// there are two "meta" bits
//   hard - specifies that the dependency producing some results for the consumer
//   real - the dependency is also translated into dependencies between passes when emitted into submission
#[derive(Clone, Copy, PartialEq, Eq)]
struct PassDependency(PackedUint<PassDependencyConfig, u32>);

// dependencies can be "hard" and "soft"
//   hard means it guards a Read After Write or Write After Write
//   soft means it guards a Write After Read
// this is important because soft dependencies do not propagate "pass is alive" status
impl PassDependency {
    fn new(pass: GraphPass, hard: bool, real: bool) -> Self {
        Self(PackedUint::new(pass.0, hard as u32 + (real as u32 * 2)))
    }
    fn is_hard(&self) -> bool {
        self.0.second() & 1 == 1
    }
    fn set_hard(&self, hard: bool) -> Self {
        Self(PackedUint::new(
            self.0.first(),
            (self.0.second() & !1) | hard as u32,
        ))
    }
    fn is_real(&self) -> bool {
        self.0.second() & 2 == 2
    }
    fn set_real(&self, real: bool) -> Self {
        Self(PackedUint::new(
            self.0.first(),
            (self.0.second() & !2) | (real as u32 * 2),
        ))
    }
    fn get_pass(&self) -> GraphPass {
        GraphPass(self.0.first())
    }
    fn index(&self) -> usize {
        self.get_pass().index()
    }
}

// submission stuff

#[derive(Clone, Copy)]
struct PassTouch {
    pass: GraphPass,
    access: vk::AccessFlags2KHR,
    stages: vk::PipelineStageFlags2KHR,
}

#[derive(Clone)]
enum ResourceState<T: ResourceMarker> {
    Uninit,
    MoveDst {
        // (count of image array layers sharing the layout, layout, the move src)
        parts: T::IfImage<SmallVec<[(PassTouch, T::IfImage<vk::ImageLayout>, GraphImage); 4]>>,
    },
    Normal {
        // TODO perhaps store whether this resource is SHARING_MODE_CONCURRENT
        layout: T::IfImage<vk::ImageLayout>,
        queue_family: u32,
        access: SmallVec<[PassTouch; 8]>,
    },
    Moved,
}

impl<T: ResourceMarker> Default for ResourceState<T> {
    fn default() -> Self {
        Self::Uninit
    }
}

// TODO pack this into 32 bits
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GraphResource {
    Image(GraphImage),
    Buffer(GraphBuffer),
}

#[derive(Clone)]
struct PassEffects {
    access: vk::AccessFlags2KHR,
    stages: vk::PipelineStageFlags2KHR,
    last_barrier: OptionalU32,
}

#[derive(Clone)]
struct OpenSubmision {
    queue: GraphQueue,
    current_pass: Option<TimelinePass>,
    pass_effects: Vec<PassEffects>,
    passes: Vec<GraphPass>,
    semaphore_dependencies: Vec<GraphSubmission>,
    barriers: Vec<(SubmissionPass, SimpleBarrier)>,
    special_barriers: Vec<(SubmissionPass, SpecialBarrier)>,
}

impl OpenSubmision {
    fn add_pass(&mut self, timeline: TimelinePass, graph: GraphPass, effects: PassEffects) {
        self.current_pass = Some(timeline);
        self.passes.push(graph);
        self.pass_effects.push(effects);
    }
    fn get_current_timeline_pass(&self) -> TimelinePass {
        self.current_pass.unwrap()
    }
    fn get_current_submission_pass(&self) -> SubmissionPass {
        SubmissionPass::new(self.passes.len().checked_sub(1).unwrap())
    }
    fn add_dependency(&mut self, dependency: GraphSubmission) {
        if !self.semaphore_dependencies.contains(&dependency) {
            self.semaphore_dependencies.push(dependency);
        }
    }
    fn finish(&mut self, queue: GraphQueue) -> Submission {
        let mut take = Submission {
            queue,
            passes: self.passes.clone(),
            semaphore_dependencies: self.semaphore_dependencies.clone(),
            barriers: self.barriers.clone(),
            special_barriers: self.special_barriers.clone(),
        };

        self.current_pass.take();
        self.pass_effects.clear();
        self.passes.clear();
        self.semaphore_dependencies.clear();
        self.barriers.clear();
        self.special_barriers.clear();

        take
    }
}

struct SubmissionRecorder<'a> {
    graph: &'a Graph,
    submissions: Vec<Submission>,
    queues: Vec<OpenSubmision>,
    current_queue: Option<(u32, GraphQueue)>,
}

impl<'a> SubmissionRecorder<'a> {
    fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            submissions: Default::default(),
            queues: Default::default(),
            current_queue: Default::default(),
        }
    }
    fn set_current_queue(&mut self, queue: GraphQueue) {
        if let Some(found) = self.queues.iter().position(|q| q.queue == queue) {
            self.current_queue = Some((found.try_into().unwrap(), queue));
        } else {
            let index = self.queues.len();
            self.queues.push(OpenSubmision {
                queue,
                current_pass: Default::default(),
                pass_effects: Default::default(),
                passes: Default::default(),
                semaphore_dependencies: Default::default(),
                barriers: Default::default(),
                special_barriers: Default::default(),
            });
            self.current_queue = Some((index.try_into().unwrap(), queue));
        }
    }
    #[inline]
    fn get_current_submission(&self) -> &OpenSubmision {
        &self.queues[self.current_queue.unwrap().0 as usize]
    }
    #[inline]
    fn get_current_submission_mut(&mut self) -> &mut OpenSubmision {
        &mut self.queues[self.current_queue.unwrap().0 as usize]
    }
    fn get_current_pass(&self) -> SubmissionPass {
        SubmissionPass::new(
            self.get_current_submission()
                .passes
                .len()
                .checked_sub(1)
                .unwrap(),
        )
    }
    fn get_closed_submission_mut(&mut self, submission: GraphSubmission) -> &mut Submission {
        &mut self.submissions[submission.index()]
    }
    fn find_queue_with_family(&self, family: u32) -> Option<GraphQueue> {
        self.queues
            .iter()
            .find(|q| self.graph.get_queue_family(q.queue) == family)
            .map(|found| found.queue)
    }
    fn add_submission_sneaky(&mut self, submission: Submission) -> GraphSubmission {
        // TODO is this neccessary?
        assert!(self.current_queue.map(|q| q.1) != Some(submission.queue));
        let index = self.submissions.len();
        self.submissions.push(submission);
        GraphSubmission::new(index)
    }
    fn begin_pass<F: FnOnce(&mut Self)>(
        &mut self,
        timeline: TimelinePass,
        graph: GraphPass,
        on_prev_end: F,
    ) -> SubmissionPass {
        let data = self.graph.get_pass_data(graph);

        // either we've started emitting passes into a different queue, or the user flushed the previously open queue
        if (self.current_queue.is_some() && self.current_queue.unwrap().1 != data.queue)
            || (!self.submissions.is_empty() && self.current_queue.is_none())
        {
            on_prev_end(self);
        }
        if self.current_queue.is_some() && self.current_queue.unwrap().1 != data.queue {
            self.__close_submission(self.current_queue.unwrap().1);
        }
        self.set_current_queue(data.queue);

        let effects = PassEffects {
            stages: data.stages.translate_special_bits(),
            access: data.access & vk::AccessFlags2KHR::WRITE_FLAGS,
            last_barrier: OptionalU32::NONE,
        };

        let submission = self.get_current_submission_mut();
        submission.add_pass(timeline, graph, effects);

        self.get_current_pass()
    }
    fn add_special_barrier(&mut self, barrier: SpecialBarrier) {
        let pass = self.get_current_pass();
        self.get_current_submission_mut()
            .special_barriers
            .push((pass, barrier));
    }
    fn add_semaphore_dependency(&mut self, dependency: GraphSubmission) {
        self.get_current_submission_mut().add_dependency(dependency);
    }
    fn add_dependency(
        &mut self,
        src_pass: GraphPass,
        src_access: vk::AccessFlags2KHR,
        src_stages: vk::PipelineStageFlags2KHR,
        dst_access: vk::AccessFlags2KHR,
        dst_stages: vk::PipelineStageFlags2KHR,
    ) {
        let (_, queue) = self.current_queue.unwrap();

        let data = self.graph.get_pass_data(src_pass);
        let meta = self.graph.get_pass_meta(src_pass);

        let scheduled = meta.scheduled_submission.get();

        // the pass is on another queue in an open submission, close it for the next step
        if data.queue != queue && scheduled.is_none() {
            let submission = self
                .__close_submission(data.queue)
                .expect("Dependency must be scheduled before the dependee");
        }

        let submission = self.get_current_submission_mut();

        // the pass has already been submitted, ignore barrier stuff and create a semaphore dependency
        if let Some(scheduled) = scheduled.get() {
            submission.add_dependency(GraphSubmission(scheduled));
            return;
        }

        // at this point we know the src is within the current submissions, so we can use a barrier
        let src = meta.scheduled_submission_position.get().get().unwrap() as usize;

        let src_pass = SubmissionPass::new(src);
        let dst_pass = submission.get_current_submission_pass();
        let effects = &mut submission.pass_effects[src];

        if src_pass == dst_pass {
            panic!("The pass cannot synchronize to itself!");
        }

        let dst_stages = dst_stages.translate_special_bits();

        let access_overlap = effects.access & src_access;
        let stages_overlap = effects.stages & src_stages.translate_special_bits();

        if !access_overlap.is_empty() || !stages_overlap.is_empty() {
            let dst_index;
            let mut barrier_index = None;
            // TODO some heuristic where we replace inefficient barriers with events?
            if let Some(barrier) = effects.last_barrier.get() {
                let (dst_end, barrier) = &mut submission.barriers[barrier as usize];
                barrier.src_stages |= stages_overlap;
                barrier.src_access |= access_overlap;
                barrier.dst_stages |= dst_stages;
                barrier.dst_access |= dst_access;

                dst_index = *dst_end;
            } else {
                let index: u32 = submission.barriers.len().try_into().unwrap();
                effects.last_barrier.set(Some(index));
                submission.barriers.push((
                    dst_pass,
                    SimpleBarrier {
                        src_stages: stages_overlap,
                        src_access: access_overlap,
                        dst_stages,
                        dst_access,
                    },
                ));
                barrier_index = Some(index);
                dst_index = dst_pass;
            }

            // TODO verify this logic, it seems flaky
            // update unsynchronized effects masks
            for effect in &mut submission.pass_effects[0..dst_index.index()] {
                if effect.access.intersects(access_overlap)
                    || effect.stages.intersects(stages_overlap)
                {
                    effect.access &= !access_overlap;
                    effect.stages &= !stages_overlap;
                    // do we want to always overwrite barriers?
                    if let Some(index) = barrier_index {
                        effect.last_barrier.set(Some(index));
                    }
                }
            }
        }
    }
    fn layout_transition(&mut self, image: GraphImage, src_dst_layout: Range<vk::ImageLayout>) {
        let submission = self.get_current_submission_mut();
        let dst_pass = submission.get_current_submission_pass();
        submission.special_barriers.push((
            dst_pass,
            SpecialBarrier::LayoutTransition {
                image,
                src_layout: src_dst_layout.start,
                dst_layout: src_dst_layout.end,
            },
        ));
    }
    fn close_current_submission(&mut self) {
        self.__close_submission(self.current_queue.unwrap().1);
        self.current_queue = None;
    }
    fn __close_submission(&mut self, queue: GraphQueue) -> Option<GraphSubmission> {
        let submission = &mut self.queues[queue.index()];
        if submission.passes.is_empty() {
            // sanity check
            assert!(submission.barriers.is_empty());
            assert!(submission.special_barriers.is_empty());
            assert!(submission.semaphore_dependencies.is_empty());

            None
        } else {
            let index: u32 = self.submissions.len().try_into().unwrap();
            let submission_index = OptionalU32::new_some(index);
            for (i, &p) in (0u32..).zip(&submission.passes) {
                let meta = self.graph.get_pass_meta(p);
                assert!(meta.scheduled_submission.get().is_none());
                meta.scheduled_submission.set(submission_index);
                meta.scheduled_submission_position
                    .set(OptionalU32::new_some(i));
            }

            let finished = submission.finish(self.current_queue.unwrap().1);
            self.submissions.push(finished);

            Some(GraphSubmission(index))
        }
    }
    fn finish(mut self) -> Vec<Submission> {
        // cleanup all open submissions, order doesn't matter since they're leaf nodes
        for queue in 0..self.queues.len() {
            self.__close_submission(GraphQueue::new(queue));
        }
        self.submissions
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
enum ResourceLastUse {
    None,
    Single(GraphSubmission, SmallVec<[SubmissionPass; 4]>),
    Multiple(SmallVec<[GraphSubmission; 4]>),
}

// a thin graph impl for the graph of queue submissions
struct SubmissionFacade<'a>(&'a Vec<Submission>, ());
impl<'a> NodeGraph for SubmissionFacade<'a> {
    type NodeData = ();

    fn node_count(&self) -> usize {
        self.0.len()
    }
    fn nodes(&self) -> Range<NodeKey> {
        0..self.0.len() as u32
    }
    fn children(&self, this: NodeKey) -> Range<ChildRelativeKey> {
        0..self.0[this as usize].semaphore_dependencies.len() as u32
    }
    fn get_child(&self, this: NodeKey, child: ChildRelativeKey) -> NodeKey {
        self.0[this as usize].semaphore_dependencies[child as usize].0
    }
    fn get_node_data(&self, this: NodeKey) -> &Self::NodeData {
        &()
    }
    fn get_node_data_mut(&mut self, this: NodeKey) -> &mut Self::NodeData {
        &mut self.1
    }
}

// TODO think of a better name
// keeps track of which resources become available in a submission
#[derive(Clone)]
pub(crate) struct SubmissionResourceReuse {
    // these are available imediatelly after a submission starts
    // because the previous users ended in its dependency submissions
    immediate: Vec<AvailabilityToken>,
    // these become available only after all of the passes have ended in the submission
    after_local_passes: Vec<(AvailabilityToken, SmallVec<[SubmissionPass; 4]>)>,
    // hashmap which holds handles of all (interned) resource lifetime ends observable from this submission
    // this is filled as passes are processed when replaying the timeline
    pub(crate) current_available_intervals: ahash::HashSet<AvailabilityToken>,
}

impl SubmissionResourceReuse {
    // merges all current_available_resources resources from dependencies which have now been closed
    fn begin(
        this: GraphSubmission,
        submissions: &[Submission],
        reuses: &mut [SubmissionResourceReuse],
    ) {
        let i = this.index();
        let mut set = std::mem::replace(
            &mut reuses[i].current_available_intervals,
            constant_ahash_hashset(),
        );

        set.insert(AvailabilityToken::NONE);
        for &d in &(&submissions[i]).semaphore_dependencies {
            set.extend(&reuses[d.index()].current_available_intervals);
        }

        reuses[i].current_available_intervals = set;
    }
}

impl Default for SubmissionResourceReuse {
    fn default() -> Self {
        Self {
            current_available_intervals: constant_ahash_hashset(),
            immediate: Vec::new(),
            after_local_passes: Vec::new(),
        }
    }
}

#[derive(Default)]
struct ResourceFirstAccess {
    closed: bool,
    accessors: SmallVec<[TimelinePass; 4]>,
    dst_layout: vk::ImageLayout,
    dst_queue_family: u32,
}

enum DeferredResourceFree {
    Image {
        vkhandle: vk::Image,
        state: ResourceMutableState<ImageMarker>,
    },
    Buffer {
        vkhandle: vk::Buffer,
        state: ResourceMutableState<BufferMarker>,
    },
}
