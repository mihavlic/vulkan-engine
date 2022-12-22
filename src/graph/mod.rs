mod reverse_edges;

use std::{
    borrow::{Borrow, Cow},
    cell::{Cell, RefCell},
    collections::{BinaryHeap, HashSet, VecDeque},
    fmt::Display,
    fs::OpenOptions,
    hash::Hash,
    io::Write,
    ops::{ControlFlow, Deref, DerefMut, Not, Range},
};

use pumice::{util::ObjectHandle, vk, VulkanResult};
use smallvec::{smallvec, SmallVec};

use crate::{
    arena::uint::{Config, OptionalU32, PackedUint},
    context::device::{Device, OwnedDevice, __test_init_device},
    graph::reverse_edges::{reverse_edges_into, ChildRelativeKey, NodeKey},
    object::{self, ImageCreateInfo, Object},
    storage::constant_ahash_randomstate,
    submission, token_abuse,
    util::{self, format_utils::Fun, macro_abuse::WeirdFormatter},
};

use self::reverse_edges::{ImmutableGraph, NodeGraph};

pub trait RenderPass: 'static {
    fn prepare(&mut self);
    fn execute(self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()>;
}

impl RenderPass for () {
    fn prepare(&mut self) {
        {}
    }
    fn execute(self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()> {
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
    fn execute(&mut self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()>;
}
impl<T: RenderPass> ObjectSafePass for StoredPass<T> {
    fn prepare(&mut self) {
        self.0.as_mut().unwrap().prepare()
    }
    fn execute(&mut self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()> {
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

struct PhysicalImageData {
    info: object::ImageCreateInfo,
    handle: vk::Image,
    state: object::ImageMutableState,
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

struct PhysicalBufferData {
    info: object::ImageCreateInfo,
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

    physical_images: Vec<PhysicalImage>,
    physical_buffers: Vec<PhysicalBufferData>,

    device: OwnedDevice,
}

impl Graph {
    pub fn new(device: OwnedDevice) -> Self {
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
            physical_images: Vec::new(),
            physical_buffers: Vec::new(),
            device,
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
                _ => {}
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
            ImageData::Transient(..) => false,
            ImageData::Imported(_) => true,
            ImageData::Swapchain(_) => true,
            ImageData::Moved(..) => unreachable!(),
        }
    }
    fn is_buffer_external(&self, mut buffer: GraphBuffer) -> bool {
        match self.get_buffer_data(buffer) {
            BufferData::Transient(..) => false,
            BufferData::Imported(_) => true,
        }
    }
    fn is_pass_alive(&self, pass: GraphPass) -> bool {
        self.pass_meta[pass.index()].alive.get()
    }
    fn get_image_data(&self, image: GraphImage) -> &ImageData {
        &self.images[image.index()]
    }
    fn get_image_data_mut(&mut self, image: GraphImage) -> &mut ImageData {
        &mut self.images[image.index()]
    }
    fn get_buffer_data(&self, buffer: GraphBuffer) -> &BufferData {
        &self.buffers[buffer.index()]
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
        }

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
        let (submissions, image_last_state, buffer_last_state) = {
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
            let mut queue_family_accesses: SmallVec<
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

            for (timeline_i, &e) in scheduled.iter().enumerate() {
                match e.get() {
                    PassEventData::Pass(p) => {
                        let on_end = |recorder: &mut SubmissionRecorder| {
                            let dst_queue_family =
                                self.get_queue_family(recorder.get_current_submission().queue);
                            for &mut (src_queue_family, ref mut src_submissions) in
                                &mut queue_family_accesses
                            {
                                use slice_group_by::GroupBy;
                                // list of submissions which we need to wait on to perform a queue ownership release
                                waitfor_submissions.clear();
                                // list of resources that will be released and then acquired
                                waitfor_resources.clear();

                                // create the queue ownership release barriers (and dummy submissions)
                                // queue ownership acquire has already been emitted during resource state updates
                                src_submissions.sort_unstable_by_key(|(_, _, _, s)| *s);
                                for entries in
                                    src_submissions.binary_group_by_key(|(_, _, _, s)| *s)
                                {
                                    // assert that only one dst pass is causing the transition
                                    debug_assert!(entries[1..]
                                        .iter()
                                        .all(|(_, _, dst, _)| *dst == entries[0].2));

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

                                    let barrier_iter = waitfor_resources.iter().map(|&res| {
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
                                        special_barriers: barrier_iter.collect(),
                                    };

                                    let sub = recorder.add_submission_sneaky(submission);
                                    recorder.add_semaphore_dependency(sub);
                                }
                            }
                            // clear the data we've just processed
                            for v in &mut queue_family_accesses {
                                v.1.clear();
                            }
                        };
                        let timeline_pass = e.to_timeline_pass().unwrap();
                        let submission_pass =
                            recorder.borrow_mut().begin_pass(timeline_pass, p, on_end);

                        let pass_position = TimelinePass::new(timeline_i);
                        let data = self.get_pass_data(p);
                        let queue = data.queue;
                        let queue_family = self.get_queue_family(queue);

                        for (i, img) in data.images.iter().enumerate() {
                            self.emit_barriers::<ImageMarker>(
                                p,
                                submission_pass,
                                queue_family,
                                img,
                                &recorder,
                                &mut *image_rw,
                                &mut queue_family_accesses,
                            );
                        }

                        for (i, buf) in data.buffers.iter().enumerate() {
                            self.emit_barriers::<BufferMarker>(
                                p,
                                submission_pass,
                                queue_family,
                                buf,
                                &recorder,
                                &mut *buffer_rw,
                                &mut queue_family_accesses,
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
                                        new_parts.extend(parts.take())
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
                                parts: TypeSome(new_parts),
                            };
                        }
                    }
                    PassEventData::Move(_) => {} // this is handled differently
                    PassEventData::Flush(q) => recorder.borrow_mut().close_current_submission(),
                }
            }

            let mut submissions = RefCell::into_inner(recorder).finish();

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

            (submissions, image_rw, buffer_rw)
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

        {
            let mut submission_children = ImmutableGraph::new();
            let graph = SubmissionFacade(&submissions, ());
            reverse_edges_into(&graph, &mut submission_children);

            let mut scratch = Vec::new();

            let mut submission_reuse =
                vec![<SubmissionResourceReuse as Default>::default(); submissions.len()];
            // TODO profile how many children submissions usually have
            // a vector may be faster here
            let mut intersection_hashmap: ahash::HashMap<GraphSubmission, u32> =
                ahash::HashMap::with_hasher(constant_ahash_randomstate());

            let image_last_touch = self.get_last_resource_usage(&image_last_state, &mut scratch);
            submission_fill_reuse::<ImageMarker>(
                image_last_touch,
                &mut submission_reuse,
                &mut intersection_hashmap,
                &submission_children,
            );
            let buffer_last_touch = self.get_last_resource_usage(&buffer_last_state, &mut scratch);
            submission_fill_reuse::<BufferMarker>(
                buffer_last_touch,
                &mut submission_reuse,
                &mut intersection_hashmap,
                &submission_children,
            );
        }

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open("target/test.dot")
            .unwrap();

        // cargo test --quiet -- graph::test_graph --nocapture && cat target/test.dot | dot -Tpng -o target/out.png
        // self.write_dot_representation(&submissions, &mut file);
        Self::write_submissions_dot_representation(&submissions, &mut file);
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
        queue_family_accesses: &mut [(
            u32,
            Vec<(
                GraphSubmission,
                SubmissionPass,
                SubmissionPass,
                GraphResource,
            )>,
        )],
    ) where
        // hack because without this the typechecker is not cooperating
        T::IfImage<vk::ImageLayout>: Copy,
    {
        let raw_resource_handle = resource_data.raw_resource_handle();
        let resource_handle = if T::IS_IMAGE {
            GraphResource::Image(GraphImage(raw_resource_handle.0))
        } else {
            GraphResource::Buffer(GraphBuffer(raw_resource_handle.0))
        };
        let &mut (ref mut state, concurrent) = &mut resource_rw[raw_resource_handle.index()];
        let dst_touch = PassTouch {
            pass,
            access: resource_data.access(),
            stages: resource_data.stages(),
        };
        let mut dst_writes = dst_touch.access.contains_write();
        let dst_layout = T::when_image(|| resource_data.start_layout());
        let normal_state = || ResourceState::Normal {
            layout: dst_layout.clone(),
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
                let dst_layout = dst_layout.take();
                let parts = parts.get_mut();
                for (part, part_i) in parts.iter_mut().zip(0u32..) {
                    let (src_touch, src_layout, move_src) = (part.0, part.1.take(), part.2);

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
                        && dst_layout != vk::ImageLayout::UNDEFINED
                        && src_queue_family != dst_queue_family
                    {
                        synchronize();
                        dst_writes = true;
                        // ownership release and acquire is handled when the current submission is closed, push some data that
                        // we want to acquire ownership from src_queue_family

                        // find the vector of pending acquires for the src queue
                        let (_, entry) = queue_family_accesses
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

                let image_undefined =
                    ImageMarker::IS_IMAGE && dst_layout.take() == vk::ImageLayout::UNDEFINED;

                // layout transition
                if !image_undefined && dst_layout.take() != src_layout.take() {
                    synchronize();
                    dst_writes = true;
                    recorder.borrow_mut().layout_transition(
                        GraphImage(raw_resource_handle.0),
                        src_layout.take()..dst_layout.take(),
                    );
                    *src_layout = dst_layout;
                }

                // queue family ownership transition
                if !image_undefined && !concurrent && *src_queue_family != dst_queue_family {
                    synchronize();
                    dst_writes = true;

                    // ownership release and acquire is handled when the current submission is closed, push some data that
                    // we want to acquire ownership from src_queue_family

                    // find the vector of pending acquires for the src queue
                    let (_, entry) = queue_family_accesses
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

fn submission_fill_reuse<T: ResourceMarker>(
    resource_last_touch: impl Iterator<Item = ResourceLastUse>,
    submission_reuse: &mut [SubmissionResourceReuse],
    intersection_hashmap: &mut ahash::HashMap<GraphSubmission, u32>,
    submission_children: &ImmutableGraph,
) {
    // we are (ab)using ResourceMarker to be generic over images
    // using branches on T::IS_IMAGE which will get constant folded
    for (i, reuse) in resource_last_touch.enumerate() {
        let handle = RawHandle::new(i);
        match reuse {
            ResourceLastUse::None => {}
            ResourceLastUse::Single(sub, passes) => {
                let sub = &mut submission_reuse[sub.index()];
                if T::IS_IMAGE {
                    sub.dead_images_local
                        .push((GraphImage::from_raw(handle), passes));
                } else {
                    sub.dead_buffers_local
                        .push((GraphBuffer::from_raw(handle), passes));
                }
            }
            ResourceLastUse::Multiple(submissions) => {
                // we need to find nodes that (directly!) depend on all of these submissions
                intersection_hashmap.clear();
                for sub in &submissions {
                    for &child in submission_children.get_children(sub.0) {
                        *intersection_hashmap
                            .entry(GraphSubmission(child))
                            .or_insert(0) += 1;
                    }
                }
                let len = submissions.len() as u32;
                for (&sub, &count) in intersection_hashmap.iter() {
                    // every submission inserted the node as its child, this means that all of the submissions share it
                    if count == len {
                        let sub = &mut submission_reuse[sub.index()];
                        if T::IS_IMAGE {
                            sub.dead_images.push(GraphImage::from_raw(handle));
                        } else {
                            sub.dead_buffers.push(GraphBuffer::from_raw(handle));
                        }
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
struct PassImageData {
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

struct PassBufferData {
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
    pass: Box<dyn ObjectSafePass>,
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
enum ImageData {
    // FIXME ImageCreateInfo is large and this scheme is weird
    Transient(object::ImageCreateInfo, Option<PhysicalImage>),
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
            ImageData::Transient(..) => "Transient",
            ImageData::Imported(_) => "Imported",
            ImageData::Swapchain(_) => "Swapchain",
            ImageData::Moved { .. } => "Moved",
        }
    }
    fn is_sharing_concurrent(&self) -> bool {
        match self {
            ImageData::Transient(info, _) => info.sharing_mode_concurrent,
            ImageData::Imported(handle) => {
                unsafe { handle.0.get_header() }
                    .info
                    .sharing_mode_concurrent
            }
            ImageData::Swapchain(_) => false,
            ImageData::Moved(..) => panic!(),
        }
    }
}
enum BufferData {
    Transient(object::BufferCreateInfo, Option<PhysicalBuffer>),
    Imported(object::Buffer),
}

impl BufferData {
    fn is_sharing_concurrent(&self) -> bool {
        match self {
            BufferData::Transient(info, _) => info.sharing_mode_concurrent,
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
    pub fn create_image(&mut self, info: impl Named<object::ImageCreateInfo>) -> GraphImage {
        let handle = GraphImage::new(self.0.images.len());
        self.0.image_meta.push(ImageMeta::new());
        self.0
            .images
            .push(info.map_to_object(|a| ImageData::Transient(a, None)));
        handle
    }
    pub fn create_buffer(&mut self, info: object::BufferCreateInfo) -> GraphBuffer {
        let handle = GraphBuffer::new(self.0.buffers.len());
        self.0.buffer_meta.push(BufferMeta::new());
        self.0
            .buffers
            .push(info.map_to_object(|a| BufferData::Transient(a, None)));
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

            let ImageData::Transient(first_info, ..) = self.0.get_image_data(images[0]).clone() else {
                invalid_data_panic(images[0], self.0.get_image_data(images[0]))
            };
            let mut first_info = first_info.clone();

            // check that all of them are transient and that they have the same format and extent
            for &i in &images[1..] {
                let data = &self.0.get_image_data(i);
                let ImageData::Transient(info, ..) = data else {
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
            }

            // update the states of the move sources
            let mut layer_offset = 0;
            for &i in &images {
                let ImageData::Transient(info, _) = self.0.get_image_data(i) else {
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

            ImageData::Transient(info, None)
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
            pass: Box::new(StoredPass(Some(pass))),
        }
    }
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

pub struct GraphImageInstance;
pub struct GraphBufferInstance;
pub struct GraphExecutor;

impl GraphExecutor {
    pub fn get_image(&self, handle: GraphImage) -> GraphImageInstance {
        todo!()
    }
    pub fn get_buffer(&self, handle: GraphBuffer) -> GraphBufferInstance {
        todo!()
    }
}

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
                fn to_raw(&self) -> RawHandle {
                    RawHandle(self.0)
                }
                #[inline]
                fn from_raw(raw: RawHandle) -> Self {
                    Self(raw.0)
                }
            }
        )+
    };
}

simple_handle! {
    RawHandle,
    pub GraphQueue, pub GraphPass, pub GraphImage, pub GraphBuffer,
    GraphSubmission, PhysicalImage, PhysicalBuffer, GraphPassMove,
    // like a GraphPassEvent but only ever points to a pass
    TimelinePass,
    // the pass in a submission
    SubmissionPass
}

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

trait TypeOption<T> {
    fn get(&self) -> &T;
    fn get_mut(&mut self) -> &mut T;
    fn take(self) -> T;
}

struct TypeSome<T>(T);
impl<T> TypeOption<T> for TypeSome<T> {
    #[inline(always)]
    fn get(&self) -> &T {
        &self.0
    }
    #[inline(always)]
    fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }
    #[inline(always)]
    fn take(self) -> T {
        self.0
    }
}
impl<T: Clone> Clone for TypeSome<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T: Copy> Copy for TypeSome<T> {}

struct TypeNone<T>(std::marker::PhantomData<fn() -> T>);
impl<T> TypeOption<T> for TypeNone<T> {
    #[inline(always)]
    fn get(&self) -> &T {
        unreachable!()
    }
    #[inline(always)]
    fn get_mut(&mut self) -> &mut T {
        unreachable!()
    }
    #[inline(always)]
    fn take(self) -> T {
        unreachable!()
    }
}
impl<T> Clone for TypeNone<T> {
    fn clone(&self) -> Self {
        Self(std::marker::PhantomData)
    }
}
impl<T> Copy for TypeNone<T> {}

trait ResourceData {
    fn access(&self) -> vk::AccessFlags2KHR;
    fn stages(&self) -> vk::PipelineStageFlags2KHR;
    fn start_layout(&self) -> vk::ImageLayout;
    fn raw_resource_handle(&self) -> RawHandle;
    fn end_layout(&self) -> Option<vk::ImageLayout>;
}

impl ResourceData for PassImageData {
    #[inline(always)]
    fn access(&self) -> vk::AccessFlags2KHR {
        self.access
    }
    #[inline(always)]
    fn stages(&self) -> vk::PipelineStageFlags2KHR {
        self.stages
    }
    #[inline(always)]
    fn start_layout(&self) -> vk::ImageLayout {
        self.start_layout
    }
    #[inline(always)]
    fn raw_resource_handle(&self) -> RawHandle {
        self.handle.to_raw()
    }
    #[inline(always)]
    fn end_layout(&self) -> Option<vk::ImageLayout> {
        self.end_layout
    }
}

impl ResourceData for PassBufferData {
    #[inline(always)]
    fn access(&self) -> vk::AccessFlags2KHR {
        self.access
    }
    #[inline(always)]
    fn stages(&self) -> vk::PipelineStageFlags2KHR {
        self.stages
    }
    #[inline(always)]
    fn start_layout(&self) -> vk::ImageLayout {
        unreachable!("This code path should never be taken!")
    }
    #[inline(always)]
    fn raw_resource_handle(&self) -> RawHandle {
        self.handle.to_raw()
    }
    #[inline(always)]
    fn end_layout(&self) -> Option<vk::ImageLayout> {
        unreachable!("This code path should never be taken!")
    }
}

trait ResourceMarker {
    const IS_IMAGE: bool;

    type IfImage<T>: TypeOption<T>;
    fn when_image<T, F: FnOnce() -> T>(fun: F) -> Self::IfImage<T>;
    type Data: ResourceData;
}

#[derive(Clone, Default)]
struct ImageMarker;
impl ResourceMarker for ImageMarker {
    const IS_IMAGE: bool = true;

    type IfImage<T> = TypeSome<T>;
    type Data = PassImageData;

    fn when_image<T, F: FnOnce() -> T>(fun: F) -> Self::IfImage<T> {
        TypeSome(fun())
    }
}

#[derive(Clone, Default)]
struct BufferMarker;
impl ResourceMarker for BufferMarker {
    const IS_IMAGE: bool = false;

    type IfImage<T> = TypeNone<T>;
    type Data = PassBufferData;

    fn when_image<T, F: FnOnce() -> T>(fun: F) -> Self::IfImage<T> {
        TypeNone(std::marker::PhantomData)
    }
}

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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum GraphResource {
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
        let take = Submission {
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
#[derive(Default, Clone)]
struct SubmissionResourceReuse {
    // these are available imediatelly after a submission starts
    // because the previous users ended in its dependency submissions
    dead_images: Vec<GraphImage>,
    dead_buffers: Vec<GraphBuffer>,
    // these become available only after all of the passes have ended in the submission
    dead_images_local: Vec<(GraphImage, SmallVec<[SubmissionPass; 4]>)>,
    dead_buffers_local: Vec<(GraphBuffer, SmallVec<[SubmissionPass; 4]>)>,
}
