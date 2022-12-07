use std::{
    borrow::{Borrow, Cow},
    cell::Cell,
    collections::{BinaryHeap, VecDeque},
    fmt::Display,
    fs::OpenOptions,
    hash::Hash,
    io::Write,
    ops::{ControlFlow, Deref, DerefMut, Not},
};

use pumice::{
    util::{impl_macros::ObjectHandle, result::VulkanResult},
    vk,
};
use smallvec::{smallvec, SmallVec};

use crate::{
    arena::uint::{Config, OptionalU32, PackedUint},
    context::device::{Device, OwnedDevice, __test_init_device},
    object::{self, ImageCreateInfo, Object},
    submission, token_abuse,
    util::{self, format_utils::Fun, macro_abuse::WeirdFormatter},
};

pub trait RenderPass: 'static {
    fn prepare(&mut self);
    fn execute(self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()>;
}

impl RenderPass for () {
    fn prepare(&mut self) {
        {}
    }
    fn execute(self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()> {
        VulkanResult::new_ok(())
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
    children_start: Cell<u32>,
    children_end: Cell<u32>,
    scheduled_position: Cell<OptionalU32>,
    scheduled_submission: Cell<OptionalU32>,
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

struct PhysicalBufferData {
    info: object::ImageCreateInfo,
}

#[derive(Clone, Default)]
struct Submission {
    passes: Vec<GraphPass>,
    dependencies: Vec<QueueSubmission>,
}

pub struct ImageSubresourceRange2 {
    pub aspect_mask: u32,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

pub struct ImageSubresourceRange3 {
    pub aspect_mask: u32,
    pub base_mip_level: u8,
    // max ~15
    pub level_count: u8,
    pub base_array_layer: u16,
    // max 2048
    pub layer_count: u16,
}

fn is_subresource_overlap(a: &ImageSubresourceRange2, b: &ImageSubresourceRange2) -> bool {
    macro_rules! range_overlap {
        ($index1:expr, $count1:expr, $index2:expr, $count2:expr) => {
            ($index1 <= ($index2 + $count2) && $index2 <= ($index1 + $count1))
        };
    }
    (a.aspect_mask & b.aspect_mask) != 0
        && range_overlap!(
            a.base_array_layer,
            a.layer_count,
            b.base_array_layer,
            b.layer_count
        )
        && range_overlap!(
            a.base_mip_level,
            a.level_count,
            b.base_mip_level,
            b.level_count
        )
}

fn is_subresource_overlap2(a: &ImageSubresourceRange3, b: &ImageSubresourceRange3) -> bool {
    macro_rules! range_overlap {
        ($index1:expr, $count1:expr, $index2:expr, $count2:expr) => {
            ($index1 <= ($index2 + $count2) && $index2 <= ($index1 + $count1))
        };
    }
    (a.aspect_mask & b.aspect_mask) != 0
        && range_overlap!(
            a.base_array_layer,
            a.layer_count,
            b.base_array_layer,
            b.layer_count
        )
        && range_overlap!(
            a.base_mip_level,
            a.level_count,
            b.base_mip_level,
            b.level_count
        )
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

    physical_images: Vec<PhysicalImage>,
    physical_buffers: Vec<PhysicalBufferData>,

    pass_children: Vec<GraphPass>,
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
            physical_images: Vec::new(),
            physical_buffers: Vec::new(),
            pass_children: Vec::new(),
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
        self.pass_children.clear();
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
                children_start: Cell::new(0),
                children_end: Cell::new(0),
                scheduled_position: Cell::new(OptionalU32::NONE),
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
                ImageData::Moved(to, _) => {
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
            BufferData::TransientRealized(_) => unreachable!(),
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
        &self.pass_children[meta.children_start.get() as usize..meta.children_end.get() as usize]
    }
    fn get_dependencies(&self, pass: GraphPass) -> &[PassDependency] {
        &self.passes[pass.index()].dependencies
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
        // either its -1 and is dead or already has been touched and has a positive number
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

        #[derive(Default, Clone, PartialEq)]
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
            // src dst
            //  W   W  -- hard
            //  W   R  -- hard
            //  R   W  -- soft
            //  R   R  -- nothing
            #[inline]
            const fn is_hard(src: bool, dst: bool) -> Option<bool> {
                match (src, dst) {
                    (true, true) => Some(true),
                    (true, false) => Some(true),
                    (false, true) => Some(false),
                    (false, false) => None,
                }
            }

            match src {
                // no dependency
                ResourceState::Uninit => {
                    *src = ResourceState::new_normal(p, dst_writing);
                }
                // inherit all dependencies
                ResourceState::MoveDst { reading, writing } => {
                    if let Some(is_hard) = is_hard(false, dst_writing) {
                        for r in reading {
                            data.try_add_dependency(*r, is_hard);
                        }
                    }
                    if let Some(is_hard) = is_hard(true, dst_writing) {
                        for r in writing {
                            data.try_add_dependency(*r, is_hard);
                        }
                    }
                }
                ResourceState::Normal { reading, writing } => {
                    if let Some(producer) = writing {
                        assert!(reading.is_empty());
                        data.try_add_dependency(
                            *producer,
                            // src is WRITE, some dependency must occur
                            is_hard(true, dst_writing).unwrap(),
                        );

                        // W W
                        if dst_writing {
                            *writing = Some(p);
                        }
                        // W R
                        else {
                            reading.push(p);
                        }
                    } else {
                        // R W
                        if dst_writing {
                            for r in &*reading {
                                data.try_add_dependency(
                                    *r,
                                    is_hard(false, /* dst_writing == */ true).unwrap(),
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
                }
                // TODO perhaps this shouldn't be a hard error and instead delegate access to the move destination
                ResourceState::Moved => panic!("Attempt to access moved resource"),
            }
        }

        let mut image_rw = vec![ResourceState::default(); self.images.len()];
        let mut buffer_rw = vec![ResourceState::default(); self.buffers.len()];

        for &e in &self.timeline {
            match e.get_kind() {
                PassEventKind::Pass => {
                    let p = e.get_pass().unwrap();
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
                PassEventKind::Move => {
                    let index = e.get_move().unwrap();
                    let ImageMove { from, to } = self.get_pass_move(index);
                    assert!(image_rw[to.index()] == ResourceState::Uninit);

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
                PassEventKind::Flush => {}
            }
        }

        // find any pass that writes to external resources, thus being considered to have side effects
        // outside of the graph and mark all of its dependencies as alive, any passes that don't get touched
        // are never scheduled and their resources never instantiated
        for (i, pass) in self.passes.iter().enumerate() {
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

        // collect the dependees of alive passes in 3 phases:
        // 1. iterate over all passes, extract their dependencies, and from this count the number of dependees (kept in the dependees_start field)
        // 2. prefix sum over the passes, allocating space in a single vector, offset into which is stored in dependees_start,
        //    dependees_end now serves as the counter of pushed dependees (the previous count in dependees_start is not needed anymore)
        // 3. iterate over all passes, extract their dependencies, and store them in the allocated space, dependees_end now points at the end
        {
            // 1.
            for p in self.get_alive_passes() {
                let pass = &self.passes[p.index()];
                for d in &pass.dependencies {
                    let dependees_start = &self.pass_meta[d.index()].children_start;
                    dependees_start.set(dependees_start.get() + 1);
                }
            }

            // 2.
            let mut offset = 0;
            for p in self.get_alive_passes() {
                // dependees_start currently stores the count of dependees
                let meta = &self.pass_meta[p.index()];
                let end = offset + meta.children_start.get();

                // start and end are the same on purpose
                meta.children_start.set(offset);
                meta.children_end.set(offset);

                offset = end;
            }

            // borrowchk woes
            let mut dependees = std::mem::take(&mut self.pass_children);
            dependees.resize(offset as usize, GraphPass::new(0));

            // 3.
            for p in self.get_alive_passes() {
                let pass = &self.passes[p.index()];
                for d in &pass.dependencies {
                    let offset = &self.pass_meta[d.index()].children_end;
                    dependees[offset.get() as usize] = p;
                    offset.set(offset.get() + 1);
                }
            }

            let _ = std::mem::replace(&mut self.pass_children, dependees);
        }

        let mut dependency_count = self
            .passes
            .iter()
            .map(|p| p.dependencies.len() as u32)
            .collect::<Vec<_>>();

        #[derive(Clone)]
        struct AvailablePass {
            pass: GraphPass,
            cost: i32,
        }

        impl AvailablePass {
            fn new(pass: GraphPass, graph: &Graph, graph_layers: &mut [i32]) -> Self {
                let layer = graph.compute_graph_layer(pass, graph_layers);
                Self { pass, cost: -layer }
            }
        }

        impl PartialEq for AvailablePass {
            fn eq(&self, other: &Self) -> bool {
                self.cost == other.cost
            }
        }
        impl PartialOrd for AvailablePass {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.cost.partial_cmp(&other.cost)
            }
        }
        impl Eq for AvailablePass {}
        impl Ord for AvailablePass {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.cost.cmp(&other.cost)
            }
        }

        let mut graph_layers = vec![0; self.passes.len()];
        let mut available: Vec<(u32, BinaryHeap<AvailablePass>)> =
            vec![(0, BinaryHeap::new()); self.queues.len()];
        let mut scheduled: Vec<GraphPass> = Vec::new();

        {
            // in a bfs, each node gets a "layer" in which is the maximum distance from a root node
            // we would like to use this in the scheduling heuristic because there are no dependencies within each layer
            // and we can saturate the gpu better
            for (pass_i, &dep_count) in dependency_count.iter().enumerate() {
                let p = GraphPass::new(pass_i);
                if !self.is_pass_alive(p) {
                    graph_layers[pass_i] = -1;
                    continue;
                }
                // root nodes get a layer 1 and are pushed into the available heaps
                if dep_count == 0 {
                    graph_layers[pass_i] = 1;
                    let queue = self.get_pass_data(p).queue;
                    let item = AvailablePass::new(p, self, &mut graph_layers);
                    available[queue.index()].1.push(item);
                }
            }

            // currently we are creating the scheduled passes by looping over each queue and poppping the locally optimal pass
            // this is rather questionable so TODO think this over
            loop {
                let len = scheduled.len();
                for queue_i in 0..self.queues.len() {
                    let (position, heap) = &mut available[queue_i];

                    let Some(AvailablePass { pass, .. }) = heap.pop() else {
                        continue;
                    };

                    self.pass_meta[pass.index()]
                        .scheduled_position
                        .set(OptionalU32::new_some(*position));

                    scheduled.push(pass);
                    *position += 1;

                    for &child in self.get_children(pass) {
                        if !self.is_pass_alive(child) {
                            continue;
                        }

                        let count = &mut dependency_count[child.index()];
                        *count -= 1;
                        if *count == 0 {
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
        }

        for (i, p) in scheduled.iter().enumerate() {
            let display = self.get_pass_display(*p);
            println!("{i} \"{display}\"");
        }

        let mut active_submissions: Vec<Submission> =
            vec![Submission::default(); self.queues.len()];
        let mut submissions = Vec::new();

        for &p in &scheduled {
            let pass = self.get_pass_data(p);
            let own = &mut active_submissions[pass.queue.index()];
            own.passes.push(p);

            for &d in &pass.dependencies {
                let dep = self.get_pass_data(d.get_pass());
                let meta = self.get_pass_meta(d.get_pass());

                if dep.queue != pass.queue {
                    let submission = &mut active_submissions[dep.queue.index()];
                    if !submission.passes.is_empty() {
                        for &p in &submission.passes {
                            let meta = self.get_pass_meta(p);
                            let index = submissions.len() as u32;
                            assert!(meta.scheduled_submission.get().is_none());
                            meta.scheduled_submission.set(OptionalU32::new_some(index));
                        }
                        let old = std::mem::replace(submission, Submission::default());
                        submissions.push((dep.queue, old));
                    }

                    let position = QueueSubmission(meta.scheduled_submission.get().get().unwrap());
                    let own = &mut active_submissions[pass.queue.index()];
                    if !own.dependencies.contains(&position) {
                        own.dependencies.push(position);
                    }
                }
            }
        }

        // flush the currently active submissions, order doesn't matter because they're leaf nodes
        for (q, s) in active_submissions.into_iter().enumerate() {
            if s.passes.is_empty() {
                continue;
            }

            let queue = GraphQueue::new(q);
            for &p in &s.passes {
                let meta = self.get_pass_meta(p);
                let index = submissions.len() as u32;
                assert!(meta.scheduled_submission.get().is_none());
                meta.scheduled_submission.set(OptionalU32::new_some(index));
            }
            submissions.push((queue, s));
        }

        // perform transitive reduction on the nodes
        // lifted from petgraph, https://docs.rs/petgraph/latest/petgraph/algo/tred/fn.dag_transitive_reduction_closure.html
        {
            let len = submissions.len();

            let mut tred = vec![Vec::new(); len];
            let mut tclos = Vec::with_capacity(len);
            let mut mark = vec![false; len];

            for i in 0..len {
                tclos.push(Vec::with_capacity(submissions[i].1.dependencies.len()));
            }

            // since each node has a list of predecessors and not successors, we need to reverse the order of all iteration
            // in relation to the petgraph implementation, it does actually work
            for i in 0..len {
                for &x in submissions[i].1.dependencies.iter().rev() {
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

            for (submission, reduced_dependencies) in submissions.iter_mut().zip(tred.into_iter()) {
                submission.1.dependencies = reduced_dependencies;
            }
        }

        #[derive(Clone, PartialEq)]
        struct SubmitLocation {
            submission: QueueSubmission,
            index: u32,
        }

        // #[derive(Clone, PartialEq)]
        // enum ImageResourceState {
        //     Unallocated,
        //     Written {
        //         queue: GraphQueue,
        //         layout: vk::ImageLayout,
        //         access: vk::AccessFlags,
        //         producer: QueueSubmission,
        //         barrier: Option<SubmitLocation>,
        //     },
        // }

        // struct PassBarrier {}

        // let mut image_state = vec![ImageResourceState::Undefined; self.images.len()];
        // let mut image_state = vec![ImageResourceState::Undefined; self.images.len()];
        // let mut out_submissions = Vec::with_capacity(submissions.len());

        // for &(q, ref s) in &submissions {
        //     for p in s.passes {}
        // }

        let mut file = OpenOptions::new()
            .write(true) // <--------- this
            .create(true)
            .truncate(true)
            .open("target/test.dot")
            .unwrap();

        // cargo test --quiet -- graph::test_graph --nocapture && cat target/test.dot | dot -Tpng -o target/out.png
        self.write_dot_representation(&submissions, &mut file);
        // Self::write_submissions_dot_representation(&submissions, &mut file);
    }
    fn write_dot_representation(
        &mut self,
        submissions: &Vec<(GraphQueue, Submission)>,
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
                    .filter(move |(qr, _)| qr.index() == queue_index)
                    .flat_map(|(_, s)| &s.passes)
                    .cloned()
            };

            // subgraph cluster_0 {
            //     style=dashed;
            //     p0[label="#0"];
            //     p1[label="#1"];
            // }
            for (i, (q, s)) in submissions.iter().enumerate() {
                let nodes = Fun::new(|w| {
                    for &p in &s.passes {
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
                    .filter_map(|(i, p)| p.map(|p| (i, p)));

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
                        writeln!(w, ";");
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
        submissions: &Vec<(GraphQueue, Submission)>,
        writer: &mut dyn std::io::Write,
    ) {
        let clusters = Fun::new(|w| {
            for (i, (q, s)) in submissions.iter().enumerate() {
                write!(w, r##"s{i}[label="#{i},{}"];"##, q.index());
                if !s.dependencies.is_empty() {
                    write!(w, "{{");
                    for p in &s.dependencies {
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

        b.add_pass_dependency(p0, p1, true);
        b.add_pass_dependency(p0, p2, true);
        b.add_pass_dependency(p2, p3, true);

        b.add_pass_dependency(p0, p4, true);
        b.add_pass_dependency(p3, p4, true);

        b.force_pass_run(p1);
        b.force_pass_run(p2);
        b.force_pass_run(p3);
        b.force_pass_run(p4);
    });
}

struct PassImageData {
    handle: GraphImage,
    usage: vk::ImageUsageFlags,
    access: vk::AccessFlags2KHR,
    start_layout: vk::ImageLayout,
    end_layout: vk::ImageLayout,
}

impl PassImageData {
    fn is_written(&self) -> bool {
        self.access.contains_write()
    }
}

struct PassBufferData {
    usage: vk::BufferUsageFlags,
    handle: GraphBuffer,
    access: vk::AccessFlags2KHR,
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
    dependencies: Vec<PassDependency>,
    pass: Box<dyn ObjectSafePass>,
}

impl PassData {
    fn try_add_dependency(&mut self, dependency: GraphPass, hard: bool) {
        if let Some(found) = self
            .dependencies
            .iter_mut()
            .find(|d| d.get_pass() == dependency)
        {
            // hard dependency overwrites a soft one
            if hard {
                found.set_hard(true);
            }
        } else {
            self.dependencies
                .push(PassDependency::new(dependency, hard));
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
    Moved(GraphImage, u32),
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
}
enum BufferData {
    Transient(object::BufferCreateInfo, Option<PhysicalBuffer>),
    TransientRealized(PhysicalBuffer),
    Imported(object::Buffer),
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
            let mut usage = vk::ImageUsageFlags::empty();
            let mut layer_offset = 0;

            let images = a.into_iter().collect::<SmallVec<_>>();

            let mut first = None;
            // check that all of them are transient and that they have the same format and extent
            for &i in &images {
                match self.0.get_image_data(i) {
                    ImageData::Transient(info, _) => {
                        usage |= info.usage;
                        if let Some(first) = first {
                            let ImageData::Transient(first, ..) = self.0.get_image_data(first) else {
                                unreachable!()
                            };

                            assert_eq!(first.size, info.size);
                            assert_eq!(first.format, info.format);
                            assert_eq!(first.samples, info.samples);
                            assert_eq!(first.tiling, info.tiling);
                        } else {
                            first = Some(i);
                        }

                        let layer_count = info.array_layers;
                        *self.0.images[i.index()].get_inner_mut() = ImageData::Moved(image, layer_offset);
                        layer_offset += layer_count;
                    }
                    // TODO perhaps it would be useful to move into already instantiated non-transient images
                    other => panic!(
                        "Only Transient images can be moved, image '{}' has state '{}'",
                        self.0.get_image_display(i),
                        other.get_variant_name()
                    ),
                }
            }

            let ImageData::Transient(first, ..) = self.0.get_image_data(first.expect("Images cannot be empty")) else {
                unreachable!()
            };

            let mut info = ImageCreateInfo {
                usage,
                array_layers: layer_offset,
                ..first.clone()
            };

            let moved = GraphPassEvent::new(PassEventKind::Move, self.0.moves.len());
            self.0.moves.push(ImageMove {
                from: images,
                to: image,
            });
            self.0.timeline.push(moved);

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
        // if the string is empty, we set the name to None
        let name = Some(name.into()).filter(|n| !n.is_empty());
        self.0.passes.push(GraphObject { name, inner: data });
        handle
    }
    pub fn add_pass_dependency(&mut self, first: GraphPass, then: GraphPass, hard: bool) {
        self.0.passes[then.index()].try_add_dependency(first, hard);
    }
    pub fn force_pass_run(&mut self, pass: GraphPass) {
        self.0.passes[pass.index()].force_run = true;
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
        PassData {
            queue: queue,
            force_run: false,
            images: self.images,
            buffers: self.buffers,
            dependencies: self.dependencies,
            pass: Box::new(StoredPass(Some(pass))),
        }
    }
    pub fn use_image(
        &mut self,
        image: GraphImage,
        usage: vk::ImageUsageFlags,
        access: vk::AccessFlags2KHR,
        start_layout: vk::ImageLayout,
        end_layout: vk::ImageLayout,
    ) {
        self.images.push(PassImageData {
            handle: image,
            usage,
            access,
            start_layout,
            end_layout,
        });
    }
    pub fn use_buffer(
        &mut self,
        buffer: GraphBuffer,
        usage: vk::BufferUsageFlags,
        access: vk::AccessFlags2KHR,
    ) {
        self.buffers.push(PassBufferData {
            handle: buffer,
            usage,
            access,
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
            #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
            $visibility struct $name(u32);
            impl $name {
                fn new(index: usize) -> Self {
                    assert!(index <= u32::MAX as usize);
                    Self(index as u32)
                }
                fn index(&self) -> usize {
                    self.0 as usize
                }
            }
        )+
    };
}

simple_handle! { pub GraphQueue, pub GraphPass, pub GraphImage, pub GraphBuffer, QueueSubmission, PhysicalImage, PhysicalBuffer, GraphPassMove }

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

enum PassEventKind {
    Pass = 0,
    Move = 1,
    Flush = 2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct GraphPassEvent(PackedUint<GraphPassEventConfig, u32>);

impl GraphPassEvent {
    fn new(kind: PassEventKind, index: usize) -> Self {
        Self(PackedUint::new(
            kind as u32,
            index.try_into().expect("Index too large"),
        ))
    }
    fn get_kind(&self) -> PassEventKind {
        match self.0.first() {
            a if a == PassEventKind::Pass as u32 => PassEventKind::Pass,
            a if a == PassEventKind::Move as u32 => PassEventKind::Move,
            a if a == PassEventKind::Flush as u32 => PassEventKind::Flush,
            _ => unreachable!(),
        }
    }
}

macro_rules! gen_pass_getters {
    ($($fun:ident, $handle:ident, $disc:expr;)+) => {
        impl GraphPassEvent {
            $(
                fn $fun (&self) -> Option<$handle> {
                    if self.0.first() == $disc as _ {
                        Some($handle(self.0.second()))
                    } else {
                        None
                    }
                }
            )+
        }
    };
}

gen_pass_getters! {
    get_pass, GraphPass, PassEventKind::Pass;
    get_move, GraphPassMove, PassEventKind::Move;
    get_flush, GraphQueue, PassEventKind::Flush;
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
    const FIRST_BITS: usize = 31;
    const SECOND_BITS: usize = 1;
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PassDependency(PackedUint<PassDependencyConfig, u32>);

// dependencies can be "hard" and "soft"
//   hard means it guards a Read After Write or Write After Write
//   soft means it guards a Write After Read
// this is important because soft dependencies do not propagate "pass is alive" status
impl PassDependency {
    fn new(pass: GraphPass, hard: bool) -> Self {
        Self(PackedUint::new(pass.0, hard as u32))
    }
    fn is_hard(&self) -> bool {
        self.0.second() == 1
    }
    fn set_hard(&self, hard: bool) -> Self {
        Self(PackedUint::new(self.0.first(), hard as u32))
    }
    fn get_pass(&self) -> GraphPass {
        GraphPass(self.0.first())
    }
    fn index(&self) -> usize {
        self.get_pass().index()
    }
}
