use std::{
    borrow::{Borrow, Cow},
    cell::Cell,
    collections::VecDeque,
    fmt::Display,
    hash::Hash,
    io::Write,
    ops::{Deref, DerefMut},
};

use pumice::{
    util::{impl_macros::ObjectHandle, result::VulkanResult},
    vk,
};
use smallvec::{smallvec, SmallVec};

use crate::{
    arena::uint::{Config, OptionalU32, PackedUint},
    context::device::{Device, __test_create_device},
    object::{self, Object},
    synchronization, token_abuse,
    util::{self, format_utils::Fun, macro_abuse::WeirdFormatter},
};

pub trait CreatePass {
    type Pass: RenderPass;
    fn create(self, builder: &mut GraphPassBuilder, device: &Device) -> Self::Pass;
}
pub trait RenderPass: 'static {
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

impl<P: RenderPass, F: FnOnce(&mut GraphPassBuilder, &Device) -> P> CreatePass for F {
    type Pass = P;
    fn create(self, builder: &mut GraphPassBuilder, device: &Device) -> Self::Pass {
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

struct PhysicalImage {
    info: object::ImageCreateInfo,
}

struct PhysicalBuffer {
    info: object::ImageCreateInfo,
}

#[derive(Clone, Default)]
struct Submission {
    passes: Vec<GraphPass>,
    dependencies: Vec<QueueSubmission>,
}

pub struct Graph {
    queues: Vec<GraphObject<synchronization::Queue>>,
    passes: Vec<GraphObject<PassData>>,
    images: Vec<GraphObject<ImageData>>,
    buffers: Vec<GraphObject<BufferData>>,

    pass_meta: Vec<PassMeta>,
    image_meta: Vec<ImageMeta>,
    buffer_meta: Vec<BufferMeta>,

    physical_images: Vec<PhysicalImage>,
    physical_buffers: Vec<PhysicalBuffer>,

    pass_children: Vec<GraphPass>,
    device: Device,
}

impl Graph {
    pub fn new(device: Device) -> Self {
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
                scheduled_position: Cell::new(OptionalU32::NONE),
                scheduled_submission: Cell::new(OptionalU32::NONE),
                queue_intervals: Cell::new(QueueIntervals::NONE),
            },
        );
        self.image_meta.resize(len, ImageMeta::new());
        self.buffer_meta.resize(len, BufferMeta::new());
    }
    fn is_image_external<'a>(&'a self, mut image: &'a GraphImage) -> bool {
        loop {
            match self.get_image_data(image) {
                ImageData::Transient(_) => break false,
                ImageData::Imported(_) => break true,
                ImageData::Swapchain(_) => break true,
                ImageData::Moved { dst, to } => {
                    image = dst;
                }
            }
        }
    }
    fn is_buffer_external(&self, mut buffer: &GraphBuffer) -> bool {
        match self.get_buffer_data(buffer) {
            BufferData::Transient(_) => false,
            BufferData::Imported(_) => true,
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

            if pass.force_run
                || pass
                    .images
                    .iter()
                    .any(|i| i.is_written() && self.is_image_external(&i.handle))
                || pass
                    .buffers
                    .iter()
                    .any(|p| p.is_written() && self.is_buffer_external(&p.handle))
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

        // we run a bfs to do a topological sort of active passes
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

        let mut active_submissions: Vec<Submission> =
            vec![Submission::default(); self.queues.len()];
        let mut submissions = Vec::new();

        for &p in &scheduled {
            let pass = self.get_pass_data(p);
            let own = &mut active_submissions[pass.queue.index()];
            own.passes.push(p);

            for &d in &pass.dependencies {
                let dep = self.get_pass_data(d);
                let meta = self.get_pass_meta(d);

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

        let mut stderr = std::io::stderr();
        self.write_dot_representation(&submissions, &mut stderr);
        // Self::write_submissions_dot_representation(&submissions, &mut stderr);
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
                    r#"q{q}[label="Queue {}:"; peripheries=0; fontsize=15; fontname="Helvetica,Arial,sans-serif bold"];"#,
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
                    let children = self.get_children(p);
                    if !children.is_empty() {
                        write!(w, "p{} -> {{", p.index());
                        for d in children {
                            write!(w, "p{} ", d.index());
                        }
                        write!(w, "}}");
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
    let device = unsafe { __test_create_device() };
    let mut g = Graph::new(device);
    g.run(|b| {
        let dummy_queue = synchronization::Queue::new(pumice::vk10::Queue::null(), 0);

        let q0 = b.import_queue(dummy_queue.clone());
        let q1 = b.import_queue(dummy_queue.clone());
        let q2 = b.import_queue(dummy_queue);

        let p0 = b.add_pass(q0, |_: &mut GraphPassBuilder, _: &Device| {});
        let p1 = b.add_pass(q0, |_: &mut GraphPassBuilder, _: &Device| {});

        let p2 = b.add_pass(q1, |_: &mut GraphPassBuilder, _: &Device| {});
        let p3 = b.add_pass(q1, |_: &mut GraphPassBuilder, _: &Device| {});

        let p4 = b.add_pass(q2, |_: &mut GraphPassBuilder, _: &Device| {});

        b.add_pass_dependency(p0, p1);
        b.add_pass_dependency(p0, p2);
        b.add_pass_dependency(p2, p3);

        b.add_pass_dependency(p0, p4);
        b.add_pass_dependency(p3, p4);

        b.force_pass_run(p1);
        b.force_pass_run(p3);
        b.force_pass_run(p4);
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
    force_run: bool,
    images: Vec<PassImageData>,
    buffers: Vec<PassBufferData>,
    dependencies: Vec<GraphPass>,
    pass: Box<dyn ObjectSafePass>,
}
enum ImageData {
    Transient(object::ImageCreateInfo),
    Imported(object::Image),
    Swapchain(object::Swapchain),
    Moved { dst: GraphImage, to: ImageMove },
}
impl ImageData {
    fn get_variant_name(&self) -> &'static str {
        match self {
            ImageData::Transient(_) => "Transient",
            ImageData::Imported(_) => "Imported",
            ImageData::Swapchain(_) => "Swapchain",
            ImageData::Moved { .. } => "Moved",
        }
    }
}
enum BufferData {
    Transient(object::BufferCreateInfo),
    Imported(object::Buffer),
}

struct GraphObject<T> {
    name: Option<Cow<'static, str>>,
    inner: T,
}

impl<T> GraphObject<T> {
    fn map_named<I, N: Named<I>, F: FnOnce(I) -> T>(named: N, fun: F) -> Self {
        let (inner, name) = named.to_named();
        let inner = fun(inner);
        Self { name, inner }
    }
    fn from_named<N: Named<T>>(named: N) -> Self {
        let (inner, name) = named.to_named();
        Self { name, inner }
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

pub struct ImageMove {
    terget_subresource: vk::ImageSubresourceLayers,
    target_offset: vk::Offset3D,
    extent: vk::Extent3D,
}

pub trait Named<T> {
    fn to_named(self) -> (T, Option<Cow<'static, str>>);
}

impl<T> Named<T> for T {
    fn to_named(self) -> (T, Option<Cow<'static, str>>) {
        (self, None)
    }
}

impl<T> Named<T> for (T, &'static str) {
    fn to_named(self) -> (T, Option<Cow<'static, str>>) {
        (self.0, Some(Cow::Borrowed(self.1)))
    }
}

impl<T> Named<T> for (T, String) {
    fn to_named(self) -> (T, Option<Cow<'static, str>>) {
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
            .push(GraphObject::map_named(swapchain, ImageData::Swapchain));
        handle
    }
    pub fn import_queue(&mut self, queue: impl Named<synchronization::Queue>) -> GraphQueue {
        let object = GraphObject::from_named(queue);
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
        self.0
            .images
            .push(GraphObject::map_named(image, ImageData::Imported));
        handle
    }
    pub fn import_buffer(&mut self, buffer: impl Named<object::Buffer>) -> GraphBuffer {
        let handle = GraphBuffer::new(self.0.buffers.len());
        self.0.buffer_meta.push(BufferMeta::new());
        self.0
            .buffers
            .push(GraphObject::map_named(buffer, BufferData::Imported));
        handle
    }
    pub fn create_image(&mut self, info: impl Named<object::ImageCreateInfo>) -> GraphImage {
        let handle = GraphImage::new(self.0.images.len());
        self.0.image_meta.push(ImageMeta::new());
        self.0
            .images
            .push(GraphObject::map_named(info, ImageData::Transient));
        handle
    }
    pub fn create_buffer(&mut self, info: object::BufferCreateInfo) -> GraphBuffer {
        let handle = GraphBuffer::new(self.0.buffers.len());
        self.0.buffer_meta.push(BufferMeta::new());
        self.0
            .buffers
            .push(GraphObject::map_named(info, BufferData::Transient));
        handle
    }
    pub fn move_image(&mut self, src: GraphImage, dst: GraphImage, to: ImageMove) {
        let src_data = &mut self.0.images[src.index()].inner;
        let ImageData::Transient(_) = src_data else {
            panic!("Only Transient images can be moved, image '{}' has state '{}'", "TODO", src_data.get_variant_name());
        };
        *src_data = ImageData::Moved { dst, to };
    }
    pub fn add_pass<T: CreatePass>(&mut self, queue: GraphQueue, pass: impl Named<T>) -> GraphPass {
        let handle = GraphPass::new(self.0.passes.len());
        let object = GraphObject::map_named(pass, |a| {
            let mut builder = GraphPassBuilder::new(self, handle);
            let pass = a.create(&mut builder, &self.0.device);
            builder.finish(queue, pass)
        });
        self.0.passes.push(object);
        handle
    }
    pub fn add_pass_dependency(&mut self, first: GraphPass, then: GraphPass) {
        self.0.passes[then.index()].dependencies.push(first);
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
    dependencies: Vec<GraphPass>,
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
    pub fn read_image(
        &mut self,
        image: GraphImage,
        access: vk::AccessFlags2KHR,
        layout: vk::ImageLayout,
    ) {
        let meta = &self.graph_builder.0.image_meta[image.index()];
        if let Some(producer) = meta.producer.get().index() {
            if producer != self.pass.index() {
                self.dependencies.push(GraphPass::new(producer));
            }
        }
        self.images.push(PassImageData {
            handle: image,
            access,
            start_layout: layout,
            end_layout: layout,
        });
    }
    pub fn read_buffer(&mut self, buffer: GraphBuffer, access: vk::AccessFlags2KHR) {
        let meta = &self.graph_builder.0.buffer_meta[buffer.index()];
        if let Some(producer) = meta.producer.get().index() {
            if producer != self.pass.index() {
                self.dependencies.push(GraphPass::new(producer));
            }
        }
        self.buffers.push(PassBufferData {
            handle: buffer,
            access,
        });
    }
    pub fn write_image(
        &mut self,
        image: GraphImage,
        access: vk::AccessFlags2KHR,
        start_layout: vk::ImageLayout,
        end_layout: vk::ImageLayout,
    ) {
        let meta = &self.graph_builder.0.image_meta[image.index()];
        if let Some(producer) = meta.producer.get().index() {
            if producer != self.pass.index() {
                self.dependencies.push(GraphPass::new(producer));
            }
        }
        // consider checking whether the access actually writes and only setting the producer then
        meta.producer
            .set(GraphPassOption::new(Some(self.pass.index())));

        self.images.push(PassImageData {
            handle: image,
            access,
            start_layout,
            end_layout,
        });
    }
    pub fn write_buffer(&mut self, buffer: GraphBuffer, access: vk::AccessFlags2KHR) {
        let meta = &self.graph_builder.0.buffer_meta[buffer.index()];
        if let Some(producer) = meta.producer.get().index() {
            if producer != self.pass.index() {
                self.dependencies.push(GraphPass::new(producer));
            }
        }
        meta.producer
            .set(GraphPassOption::new(Some(self.pass.index())));

        self.buffers.push(PassBufferData {
            handle: buffer,
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

simple_handle! { pub GraphQueue, pub GraphPass, pub GraphImage, pub GraphBuffer, QueueSubmission }

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
