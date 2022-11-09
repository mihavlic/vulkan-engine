use std::{
    borrow::Cow,
    cell::Cell,
    collections::VecDeque,
    hash::Hash,
    ops::{Deref, DerefMut},
};

use pumice::{
    util::{impl_macros::ObjectHandle, result::VulkanResult},
    vk,
};
use smallvec::{smallvec, SmallVec};

use crate::{
    arena::uint::{Config, OptionalU32, PackedUint},
    context::device::Device,
    object::{self, Object},
    synchronization,
};

pub trait CreatePass<'a> {
    type Pass: RenderPass;
    fn create(self, builder: &'a mut GraphPassBuilder, device: &'a Device) -> Self::Pass;
}
pub trait RenderPass {
    fn prepare(&mut self);
    fn execute(self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()>;
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

impl<'a, P: RenderPass, F: FnOnce(&'a mut GraphPassBuilder, &'a Device) -> P> CreatePass<'a> for F {
    type Pass = P;
    fn create(self, builder: &'a mut GraphPassBuilder, device: &'a Device) -> Self::Pass {
        self(builder, device)
    }
}

impl RenderPass for () {
    fn prepare(&mut self) {
        {}
    }
    fn execute(self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()> {
        VulkanResult::new_ok(())
    }
}

#[derive(Clone)]
struct PassMeta {
    alive: Cell<bool>,
    children_start: Cell<u32>,
    children_end: Cell<u32>,
    // // numeric interval for answering "is this node a descendant of some other node?"
    // // I have no idea what the method is called
    // interval_min: Cell<u32>,
    // interval_max: Cell<u32>,
    scheduled_position: Cell<OptionalU32>,
    queue_intervals: Cell<QueueIntervals>,
}
#[derive(Clone)]
struct ImageMeta {
    alive: Cell<bool>,
    physical: OptionalU32,
}
#[derive(Clone)]
struct BufferMeta {
    alive: Cell<bool>,
    physical: OptionalU32,
}

struct PhysicalImage {
    location: u64,
    info: object::ImageCreateInfo,
}

struct PhysicalBuffer {
    location: u64,
    info: object::ImageCreateInfo,
}

pub struct Graph {
    queues: Vec<synchronization::Queue>,
    passes: Vec<GraphObject<PassData>>,
    images: Vec<GraphObject<ImageData>>,
    buffers: Vec<GraphObject<BufferData>>,

    pass_meta: Vec<PassMeta>,
    image_meta: Vec<ImageMeta>,
    buffer_meta: Vec<BufferMeta>,

    physical_images: Vec<PhysicalImage>,
    physical_buffers: Vec<PhysicalBuffer>,

    pass_children: Vec<GraphPass>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            queues: Vec::new(),
            passes: Vec::new(),
            images: Vec::new(),
            buffers: Vec::new(),
            pass_meta: Vec::new(),
            image_meta: Vec::new(),
            buffer_meta: Vec::new(),
            physical_images: Vec::new(),
            physical_buffers: Vec::new(),
            pass_children: Vec::new(),
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
        pass.dependencies
            .iter()
            .for_each(|&p| self.mark_pass_alive(p));
    }
    fn mark_image_alive(&self, mut handle: GraphImage) {
        loop {
            let i = handle.index() as usize;
            self.image_meta[i].alive.set(true);
            match &*self.images[i] {
                ImageData::Moved { dst, .. } => {
                    handle = *dst;
                }
                _ => {}
            }
        }
    }
    fn mark_buffer_alive(&self, handle: GraphBuffer) {
        let i = handle.index() as usize;
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
                // interval_min: Cell::new(0),
                // interval_max: Cell::new(0),
                scheduled_position: Cell::new(OptionalU32::NONE),
                queue_intervals: Cell::new(QueueIntervals::NONE),
            },
        );
        self.image_meta.resize(
            len,
            ImageMeta {
                alive: Cell::new(false),
                physical: OptionalU32::NONE,
            },
        );
        self.buffer_meta.resize(
            len,
            BufferMeta {
                alive: Cell::new(false),
                physical: OptionalU32::NONE,
            },
        );
    }
    fn is_image_external<'a>(&'a self, mut image: &'a GraphImage) -> bool {
        loop {
            match self.get_image_data(image) {
                ImageData::Transient { .. } => break false,
                ImageData::Imported { .. } => break true,
                ImageData::Swapchain { .. } => break true,
                ImageData::Moved { dst, to } => {
                    image = dst;
                }
            }
        }
    }
    fn is_buffer_external(&self, mut buffer: &GraphBuffer) -> bool {
        match self.get_buffer_data(buffer) {
            BufferData::Transient { .. } => false,
            BufferData::Imported { .. } => true,
        }
    }
    fn is_pass_alive(&self, pass: GraphPass) -> bool {
        self.pass_meta[pass.index()].alive.get()
    }
    fn get_image_data(&self, image: &GraphImage) -> &ImageData {
        &self.images[image.index() as usize]
    }
    fn get_buffer_data(&self, buffer: &GraphBuffer) -> &BufferData {
        &self.buffers[buffer.index() as usize]
    }
    fn get_pass_data(&self, pass: GraphPass) -> &PassData {
        &self.passes[pass.0 as usize]
    }
    fn get_pass_meta(&self, pass: GraphPass) -> &PassMeta {
        &self.pass_meta[pass.0 as usize]
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
    fn get_dependencies(&self, pass: GraphPass) -> &[GraphPass] {
        &self.passes[pass.index()].dependencies
    }
    // fn build_interval_tree(&self, p: GraphPass, i: &mut u32) -> (u32, u32) {
    //     let meta = &self.pass_meta[p.index()];
    //     if meta.interval_min.get() > 0 {
    //         return (meta.interval_min.get(), meta.interval_max.get());
    //     }

    //     let children = self.get_children(p);

    //     let mut min = u32::MAX;
    //     let mut max = u32::MIN;

    //     if children.is_empty() {
    //         *i += 1;
    //         min = *i;
    //         max = *i;
    //     } else {
    //         for &d in children {
    //             let (a, b) = self.build_interval_tree(d, i);
    //             min = a.min(min);
    //             max = b.max(max);
    //         }
    //     }

    //     meta.interval_min.set(min);
    //     meta.interval_max.set(max);

    //     (min, max)
    // }
    pub fn run<F: FnOnce(&mut GraphBuilder)>(&mut self, fun: F) {
        self.clear();

        // get the graph from the user
        // sound because GraphBuilder is repr(transparent)
        let builder = unsafe { std::mem::transmute::<&mut Graph, &mut GraphBuilder>(self) };
        fun(builder);
        self.prepare_meta();

        // find any pass that writes to external resources, thus being considered to have side effects
        // outside of the graph and mark all of its dependencies as alive, any passes that don't get touched
        // are never scheduled and their resources never instantiated
        for (i, pass) in self.passes.iter().enumerate() {
            if self.pass_meta[i].alive.get() {
                continue;
            }

            if true
            // pass
            //     .images
            //     .iter()
            //     .any(|i| i.is_written() && self.is_image_external(&i.handle))
            //     || pass
            //         .buffers
            //         .iter()
            //         .any(|p| p.is_written() && self.is_buffer_external(&p.handle))
            {
                self.mark_pass_alive(GraphPass(i as u32));
                for i in &pass.images {
                    self.mark_image_alive(i.handle);
                }
                for b in &pass.buffers {
                    self.mark_buffer_alive(b.handle);
                }
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
            let mut dependees = std::mem::replace(&mut self.pass_children, Vec::new());
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

        // // create a weird interval "tree", it will be used for answering "is this node on the path to another node?"
        // // please note that due to this actually being an arbitrary collection of graphs this can give false positives (TODO clarify)
        // //
        // //      1*2 2*2  3*3
        // //      / \ /     |
        // //    1*1 2*2    3*3
        // //     |   |      |
        // //    1*1 2*2    3*3

        // let mut i = 0;
        // for p in self.get_alive_passes() {
        //     self.build_interval_tree(p, &mut i);
        // }

        // we run a bfs to do a topological sort of active passes
        // this will result in a submission order where neighboring passes have
        // no dependencies so that the gpu can be saturated

        // TODO use some heuristics for scheduling

        let mut queue_positions: Vec<u32> = vec![0; self.queues.len()];
        let mut scheduled: Vec<GraphPass> = Vec::new();
        let mut dependency_count = self
            .passes
            .iter()
            .map(|p| p.dependencies.len() as u32)
            .collect::<Vec<_>>();

        let mut stacks: Vec<VecDeque<GraphPass>> = vec![VecDeque::new(); self.queues.len()];
        for (i, &dep_count) in dependency_count.iter().enumerate() {
            let p = GraphPass::new(i);
            if dep_count == 0 && self.is_pass_alive(p) {
                let queue = self.get_pass_data(p).queue;
                stacks[queue.index()].push_front(p);
            }
        }

        'queues: loop {
            let len = scheduled.len();
            for queue_i in 0..self.queues.len() {
                let stack = &mut stacks[queue_i];

                // TODO use some actual heuristic for scheduling
                let Some(next) = stack.pop_back() else {
                    continue;
                };

                // we have now decided the final relative order of the pass, set it
                self.pass_meta[next.index()]
                    .scheduled_position
                    .set(OptionalU32::new_some(queue_positions[queue_i]));

                scheduled.push(next);
                queue_positions[queue_i] += 1;

                for &child in self.get_children(next) {
                    if !self.is_pass_alive(child) {
                        continue;
                    }

                    let count = &mut dependency_count[child.index()];
                    *count -= 1;
                    if *count == 0 {
                        stacks[self.get_pass_data(child).queue.index()].push_front(child);
                    }
                }
            }

            // if the length is unchanged we have exhausted all passes
            if len == scheduled.len() {
                break;
            }
        }

        // passes execute on multiple queues
        //
        //  A     B      B[2] depends on A[2], A[4] depends on B[4]
        // [1]   [1]     This creates points of forced `submission`/`start` of command buffer to allow us to signal the neccessary semaphores.
        // [2]-->[2]     We can move `submission` points down and `begin` points up and also merge them, as long as they don't cross over each other
        // [3]   [3]
        // [4]<--[4]
        // [5]   [5]

        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        enum QueueDelimiter {
            Begin,
            Submit,
        }

        let mut queue_delimiters: Vec<Vec<(u32, QueueDelimiter)>> =
            vec![vec![(0, QueueDelimiter::Begin)]; self.queues.len()];

        for &p in &scheduled {
            let pass = self.get_pass_data(p);
            let queue = pass.queue;

            for &p in &pass.dependencies {
                let data = self.get_pass_data(p);
                let meta = self.get_pass_meta(p);

                if data.queue != pass.queue {
                    queue_delimiters[data.queue.index()].push((
                        meta.scheduled_position.get().get().unwrap(),
                        QueueDelimiter::Submit,
                    ));
                }
            }

            for &p in self.get_children(p) {
                let data = self.get_pass_data(p);
                let meta = self.get_pass_meta(p);

                if data.queue != pass.queue {
                    queue_delimiters[data.queue.index()].push((
                        meta.scheduled_position.get().get().unwrap(),
                        QueueDelimiter::Begin,
                    ));
                }
            }
        }

        for queue_i in 0..self.queues.len() {
            queue_delimiters[queue_i].push((
                queue_positions[queue_i].saturating_sub(1),
                QueueDelimiter::Submit,
            ));
        }

        for delimiters in &mut queue_delimiters {
            // delimiters.dedup_by_key(|(_, d)| *d);
        }

        for q in 0..self.queues.len() {
            let mut passes = scheduled
                .iter()
                .filter(|&&p| self.get_pass_data(p).queue.index() == q);
            let mut delimiters = queue_delimiters[q].iter();

            while let Some(&next) = passes.next() {
                let data = self.get_pass_data(next);
                let meta = self.get_pass_meta(next);
                let pass_pos = meta.scheduled_position.get().get().unwrap();

                let mut before = "";
                let mut after = "";

                while let Some(&(pos, kind)) = delimiters.clone().next() {
                    if pass_pos == pos {
                        delimiters.next();
                        match kind {
                            QueueDelimiter::Begin => before = "<begin> ",
                            QueueDelimiter::Submit => after = " <submit>",
                        }
                    } else {
                        break;
                    }
                }

                print!("{before}#{}{after} ", next.index());
            }

            println!();
        }
    }
    fn build_queue_intervals(
        &self,
        p: GraphPass,
        queue_intervals: &mut Vec<Cell<u32>>,
    ) -> QueueIntervals {
        let pass = &self.passes[p.index()];
        let meta = &self.pass_meta[p.index()];
        let queue_position = meta.queue_intervals.get();

        if queue_position.index().is_some() {
            queue_position
        } else {
            let queue = pass.queue.index();
            let count = self.queues.len();
            let end = queue_intervals.len();
            queue_intervals.extend((0..count).map(|_| Cell::new(0u32)));

            let pass_prev_position = meta
                .scheduled_position
                .get()
                .get()
                .unwrap()
                .saturating_sub(1);
            queue_intervals[end + queue].set(pass_prev_position);

            for &d in self.get_dependencies(p) {
                let d_pass = self.get_pass_data(d);
                let d_meta = &self.pass_meta[d.index()];
                queue_intervals[end + d_pass.queue.index()].set(
                    queue_intervals[end + d_pass.queue.index()]
                        .get()
                        .max(d_meta.scheduled_position.get().get().unwrap()),
                );

                let dep_intervals = self.build_queue_intervals(d, queue_intervals);

                let mut i = end;
                let mut j = dep_intervals.index().unwrap();

                for _ in 0..count {
                    queue_intervals[i].set(queue_intervals[i].get().max(queue_intervals[j].get()));
                    i += 1;
                    j += 1;
                }
            }

            // the closest position on the pass' queue that can be achieved is the previous pass
            assert_eq!(queue_intervals[end + queue].get(), pass_prev_position);

            let intervals = QueueIntervals::new(Some(end));
            meta.queue_intervals.set(intervals);
            intervals
        }
    }
}

#[test]
fn test_graph() {
    let mut g = Graph::new();
    g.run(|b| {
        let q = b.import_queue(synchronization::Queue::new(pumice::vk10::Queue::null(), 0));
        let r = b.import_queue(synchronization::Queue::new(pumice::vk10::Queue::null(), 0));

        let p1 = b.add_pass(q, |_, _| {});
        let p2 = b.add_pass(q, |_, _| {});

        let s1 = b.add_pass(r, |_, _| {});
        let s2 = b.add_pass(r, |_, _| {});

        b.add_pass_dependency(p1, p2);
        b.add_pass_dependency(p1, s2);
    });
}

struct PassImageData {
    handle: GraphImage,
    access: vk::AccessFlags2KHR,
    start_layout: vk::ImageLayout,
    end_layout: vk::ImageLayout,
}

impl PassImageData {
    fn is_written(&self) -> bool {
        self.access.contains_write_flag()
    }
}

struct PassBufferData {
    handle: GraphBuffer,
    access: vk::AccessFlags2KHR,
}

impl PassBufferData {
    fn is_written(&self) -> bool {
        self.access.contains_write_flag()
    }
}

struct PassData {
    queue: GraphQueue,
    images: Vec<PassImageData>,
    buffers: Vec<PassBufferData>,
    dependencies: Vec<GraphPass>,
    pass: Box<dyn ObjectSafePass>,
}
enum ImageData {
    Transient { info: object::ImageCreateInfo },
    Imported { handle: object::Image },
    Swapchain { handle: object::Swapchain },
    Moved { dst: GraphImage, to: ImageMove },
}
enum BufferData {
    Transient { info: object::BufferCreateInfo },
    Imported { handle: object::Buffer },
}

struct GraphObject<T> {
    name: Option<Cow<'static, str>>,
    inner: T,
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

pub struct ImageMove {
    terget_subresource: vk::ImageSubresourceLayers,
    target_offset: vk::Offset3D,
    extent: vk::Extent3D,
}

#[repr(transparent)]
pub struct GraphBuilder(Graph);
impl GraphBuilder {
    pub fn acquire_swapchain(&mut self, swapchain: object::Swapchain) -> GraphImage {
        todo!()
    }
    pub fn import_queue(&mut self, queue: synchronization::Queue) -> GraphQueue {
        let handle = GraphQueue::new(self.0.queues.len());
        self.0.queues.push(queue);
        handle
    }
    pub fn import_image(&mut self, image: object::Image) -> GraphImage {
        todo!()
    }
    pub fn import_buffer(&mut self, buffer: object::Buffer) -> GraphBuffer {
        todo!()
    }
    pub fn create_image(&mut self, info: object::ImageCreateInfo) -> GraphImage {
        todo!()
    }
    pub fn create_buffer(&mut self, info: object::BufferCreateInfo) -> GraphBuffer {
        todo!()
    }
    pub fn move_image(&mut self, src: &GraphImage, dst: &GraphImage, to: ImageMove) {
        todo!()
    }
    pub fn add_pass<'a, T: CreatePass<'a>>(&mut self, queue: GraphQueue, pass: T) -> GraphPass {
        let handle = GraphPass::new(self.0.passes.len());
        self.0.passes.push(GraphObject {
            name: None,
            inner: PassData {
                queue,
                images: Vec::new(),
                buffers: Vec::new(),
                dependencies: Vec::new(),
                pass: Box::new(StoredPass(Some(()))),
            },
        });
        handle
    }
    pub fn add_pass_dependency(&mut self, first: GraphPass, then: GraphPass) {
        self.0.passes[then.index()].dependencies.push(first);
    }
}

pub struct GraphPassBuilder(PassData);
impl GraphPassBuilder {
    pub fn read_image(
        &mut self,
        image: &GraphImage,
        access: vk::AccessFlags2KHR,
        layout: vk::ImageLayout,
    ) {
    }
    pub fn read_buffer(&mut self, buffer: &GraphBuffer, access: vk::AccessFlags2KHR) {}
    pub fn write_image(
        &mut self,
        image: GraphImage,
        access: vk::AccessFlags2KHR,
        start_layout: vk::ImageLayout,
        end_layout: vk::ImageLayout,
    ) -> GraphImage {
        todo!()
    }
    pub fn write_buffer(
        &mut self,
        buffer: GraphBuffer,
        access: vk::AccessFlags2KHR,
    ) -> GraphBuffer {
        todo!()
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
    ($($name:ident),+) => {
        $(
            #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
            pub struct $name(u32);
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

simple_handle! { GraphQueue, GraphPass, GraphImage, GraphBuffer }

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

optional_index! { QueueIntervals }

#[derive(Clone, Copy)]
struct QueuePositionConfig;
impl Config for QueuePositionConfig {
    // render pass index
    const FIRST_BITS: usize = 24;
    // executing queue index (graph-wise index, not device)
    const SECOND_BITS: usize = 8;
}

#[derive(Clone, Copy)]
struct QueuePosition(PackedUint<QueuePositionConfig, u32>);
impl QueuePosition {
    #[inline]
    fn new(position: u32, queue: GraphQueue) -> Self {
        Self(PackedUint::new(position, queue.0))
    }
    #[inline]
    fn position(&self) -> GraphPass {
        GraphPass(self.0.first())
    }
    #[inline]
    fn queue(&self) -> GraphQueue {
        GraphQueue(self.0.second())
    }
}

// macro_rules! resource_handle {
//     ($($name:ident),+) => {
//         $(
//             pub struct $name(PassResourceIndex);
//             impl $name {
//                 fn new(resource: u32, producer: Option<u32>) -> Self {
//                     Self(PassResourceIndex::new(resource, producer))
//                 }
//                 fn resource(&self) -> u32 {
//                     self.0.resource()
//                 }
//                 fn producer(&self) -> Option<GraphPass> {
//                     self.0.producer().map(GraphPass)
//                 }
//                 /// creates a copy of the handle, bypassing the usual producer-consumer invariants
//                 fn clone_internal(&self) -> Self {
//                     Self(self.0)
//                 }
//             }
//         )+
//     };
// }

// resource_handle! { GraphImage, GraphBuffer }
