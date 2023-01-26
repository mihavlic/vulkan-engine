use super::{
    allocator::{AvailabilityToken, Suballocator},
    execute::{CompiledGraph, LegacySemaphoreStack, PhysicalBufferData, PhysicalImageData},
    record::{
        BufferData, CompilationInput, GraphBuilder, ImageData, ImageMove, MovedImageEntry, PassData,
    },
    resource_marker::{ResourceData, ResourceMarker, TypeSome},
    reverse_edges::{ChildRelativeKey, DFSCommand, ImmutableGraph, NodeGraph, NodeKey},
    task::{
        CompileGraphicsPipelinesTask, ComputePipelinePromise, ExecuteFnTask,
        GraphicsPipelinePromise, GraphicsPipelineResult, Promise, SendAny,
    },
    GraphBuffer, GraphImage, GraphObject, GraphObjectDisplay, GraphPass, GraphPassMove, GraphQueue,
    GraphSubmission, ObjectSafeCreatePass, PhysicalBuffer, PhysicalImage, RawHandle,
    SubmissionPass, TimelinePass,
};
use crate::{
    arena::uint::{Config, OptionalU32, PackedUint},
    device::{
        batch::GenerationId,
        submission::{self, QueueSubmission},
        Device, OwnedDevice,
    },
    graph::{
        allocator::MemoryKind,
        execute::{CompiledGraphState, DeferredResourceFree, GraphExecutor},
        resource_marker::{BufferMarker, ImageMarker, TypeOption},
        reverse_edges::reverse_edges_into,
        task::GraphicsPipelineSrc,
    },
    object::{
        self, raw_info_handle_renderpass, BufferMutableState, ConcreteGraphicsPipeline,
        GetPipelineResult, GraphicsPipeline, ImageMutableState, RenderPassMode,
        SwapchainAcquireStatus, SynchronizeResult,
    },
    passes::RenderPass,
    storage::{constant_ahash_hashmap, constant_ahash_hashset},
};
use ahash::HashMap;
use bumpalo::Bump;
use fixedbitset::FixedBitSet;
use parking_lot::lock_api::RawRwLock;
use pumice::{vk, vk10::CommandPoolCreateInfo, DeviceWrapper};
use rayon::ThreadPool;
use smallvec::{smallvec, SmallVec};
use std::{
    borrow::Cow,
    cell::{Cell, RefCell, RefMut},
    collections::{hash_map::Entry, BinaryHeap},
    fmt::Display,
    mem::ManuallyDrop,
    ops::{Deref, Range},
};

#[derive(Clone)]
pub(crate) struct PassMeta {
    scheduled_submission: Cell<OptionalU32>,
    scheduled_submission_position: Cell<OptionalU32>,
}
#[derive(Clone)]
pub(crate) struct ImageMeta {
    pub(crate) concurrent: bool,
}
impl ImageMeta {
    pub(crate) fn new() -> Self {
        Self { concurrent: false }
    }
}
#[derive(Clone)]
pub(crate) struct BufferMeta {
    pub(crate) concurrent: bool,
}
impl BufferMeta {
    pub(crate) fn new() -> Self {
        Self { concurrent: false }
    }
}

pub(crate) enum PassObjectState {
    Initial(Box<dyn ObjectSafeCreatePass>),
    Created(Box<dyn RenderPass>),
    DummyNone,
}

#[derive(Clone)]
pub(crate) struct SimpleBarrier {
    pub src_stages: vk::PipelineStageFlags2KHR,
    pub src_access: vk::AccessFlags2KHR,
    pub dst_stages: vk::PipelineStageFlags2KHR,
    pub dst_access: vk::AccessFlags2KHR,
}

#[derive(Clone)]
pub(crate) enum SpecialBarrier {
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
pub(crate) struct Submission {
    pub(crate) queue: GraphQueue,
    pub(crate) passes: Vec<GraphPass>,
    pub(crate) semaphore_dependencies: Vec<GraphSubmission>,
    // barriers and special_barriers may have SubmissionPass targets that go beyond the actual passes vector
    // these are emitted to fixup synchronization for queue ownership transfer
    pub(crate) barriers: Vec<(SubmissionPass, SimpleBarrier)>,
    pub(crate) special_barriers: Vec<(SubmissionPass, SpecialBarrier)>,
    // flag tracking whether this submission has a final barrier which synchronizes against all stages and accesses
    // this is used for ownership transitions and swapchain presentation
    pub(crate) contains_end_all_barrier: bool,
}

impl Submission {
    // the SubmissionPass index of the "end all barrier"
    const END_ALL_BARRIER_SUBMISSION_PASS: SubmissionPass = SubmissionPass(u32::MAX - 1);
    // the SubmissionPass index of all special barriers after the "end all barrier"
    const AFTER_END_ALL_BARRIER_SUBMISSION_PASS: SubmissionPass = SubmissionPass(u32::MAX);
    // adds a special barrier that happens after all the passes in the submission have finished
    // useful for resource queue ownership transfers
    pub(crate) fn add_final_special_barrier(&mut self, barrier: SpecialBarrier) {
        if !self.contains_end_all_barrier {
            self.barriers.push((
                Self::END_ALL_BARRIER_SUBMISSION_PASS,
                SimpleBarrier {
                    src_stages: vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                    src_access: vk::AccessFlags2KHR::all(),
                    dst_stages: vk::PipelineStageFlags2KHR::empty(),
                    dst_access: vk::AccessFlags2KHR::empty(),
                },
            ));
            self.contains_end_all_barrier = true;
        }
        self.special_barriers
            .push((Self::AFTER_END_ALL_BARRIER_SUBMISSION_PASS, barrier));
    }
}

pub struct GraphContext {
    graphics_pipelines: Vec<(GraphicsPipeline, Vec<GraphicsPipelineResult>)>,
    function_promises: Vec<Option<SendAny>>,
}

impl GraphContext {
    pub fn resolve_graphics_pipeline(
        &mut self,
        promise: GraphicsPipelinePromise,
    ) -> ConcreteGraphicsPipeline {
        let GraphicsPipelinePromise {
            batch_index,
            mode_offset,
        } = promise;

        let (archandle, entries) = &mut self.graphics_pipelines[batch_index as usize];
        let entry = &mut entries[mode_offset as usize];
        let concrete = match entry {
            GraphicsPipelineResult::Compile(_) => unreachable!(),
            GraphicsPipelineResult::CompiledFinal(handle) => *handle,
            GraphicsPipelineResult::Wait(lock, mode_hash) => {
                lock.lock_shared();

                let result = unsafe {
                    archandle
                        .0
                        .access_mutable(|d| &d.mutable, |m| m.get_pipeline(*mode_hash, || panic!()))
                };

                let handle = match result {
                    GetPipelineResult::Ready(handle) => handle,
                    GetPipelineResult::Promised(_) => unreachable!(),
                    GetPipelineResult::MustCreate(_) => unreachable!(),
                };

                *entry = GraphicsPipelineResult::Ready(handle);
                handle
            }
            GraphicsPipelineResult::Ready(handle) => *handle,
        };

        ConcreteGraphicsPipeline(archandle.clone(), concrete)
    }
    pub fn resolve_compute_pipeline(&mut self, promise: ComputePipelinePromise) -> () {
        todo!()
    }
    pub fn resolve_promise<T: 'static>(&mut self, promise: Promise<T>) -> T {
        *self.function_promises[promise.0.index()]
            .take()
            .unwrap()
            .into_any()
            .downcast()
            .unwrap()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphPassEventConfig;
impl Config for GraphPassEventConfig {
    const FIRST_BITS: usize = 2;
    const SECOND_BITS: usize = 30;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PassEventData {
    Pass(GraphPass),
    Move(GraphPassMove),
    Flush(GraphQueue),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphPassEvent(PackedUint<GraphPassEventConfig, u32>);

impl GraphPassEvent {
    pub(crate) fn to_timeline_pass(self) -> Option<TimelinePass> {
        self.get_pass().map(|p| TimelinePass(p.0))
    }
}

macro_rules! gen_pass_getters {
    ($name:ident, $data:ident: $($fun:ident, $handle:ident, $disc:tt: $val:expr;)+) => {
        #[allow(unused)]
        impl $name {
            pub(crate) fn new(data: $data) -> Self {
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
                pub(crate) fn $fun (&self) -> Option<$handle> {
                    if self.0.first() == $val {
                        Some($handle(self.0.second()))
                    } else {
                        None
                    }
                }
            )+
            pub(crate) fn get(&self) -> $data {
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CombinedResourceConfig;
impl Config for CombinedResourceConfig {
    const FIRST_BITS: usize = 1;
    const SECOND_BITS: usize = 31;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum GraphResource {
    Image(GraphImage),
    Buffer(GraphBuffer),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CombinedResourceHandle(PackedUint<CombinedResourceConfig, u32>);

impl CombinedResourceHandle {
    pub(crate) fn new_image(image: GraphImage) -> Self {
        Self(PackedUint::new(0, image.0))
    }
    pub(crate) fn new_buffer(buffer: GraphBuffer) -> Self {
        Self(PackedUint::new(1, buffer.0))
    }
    pub(crate) fn get_image(self) -> Option<GraphPass> {
        if self.0.first() == 0 {
            Some(GraphPass(self.0.second()))
        } else {
            None
        }
    }
    pub(crate) fn get_buffer(self) -> Option<GraphPass> {
        if self.0.first() == 1 {
            Some(GraphPass(self.0.second()))
        } else {
            None
        }
    }
    pub(crate) fn unpack(self) -> GraphResource {
        let second = self.0.second();
        match self.0.first() {
            0 => GraphResource::Image(GraphImage(second)),
            1 => GraphResource::Buffer(GraphBuffer(second)),
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct PassDependencyConfig;
impl Config for PassDependencyConfig {
    const FIRST_BITS: usize = 30;
    const SECOND_BITS: usize = 2;
}

// there are two "meta" bits
//   hard - specifies that the dependency producing some results for the consumer
//   real - the dependency is also translated into dependencies between passes when emitted into submission
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct PassDependency(PackedUint<PassDependencyConfig, u32>);

// dependencies can be "hard" and "soft"
//   hard means it guards a Read After Write or Write After Write
//   soft means it guards a Write After Read
// this is important because soft dependencies do not propagate "pass is alive" status
impl PassDependency {
    pub(crate) fn new(pass: GraphPass, hard: bool, real: bool) -> Self {
        Self(PackedUint::new(pass.0, hard as u32 + (real as u32 * 2)))
    }
    pub(crate) fn is_hard(&self) -> bool {
        self.0.second() & 1 == 1
    }
    pub(crate) fn set_hard(&mut self, hard: bool) {
        *self = Self(PackedUint::new(
            self.0.first(),
            (self.0.second() & !1) | hard as u32,
        ));
    }
    pub(crate) fn is_real(&self) -> bool {
        self.0.second() & 2 == 2
    }
    pub(crate) fn set_real(&mut self, real: bool) {
        *self = Self(PackedUint::new(
            self.0.first(),
            (self.0.second() & !2) | (real as u32 * 2),
        ));
    }
    pub(crate) fn get_pass(&self) -> GraphPass {
        GraphPass(self.0.first())
    }
    pub(crate) fn index(&self) -> usize {
        self.get_pass().index()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PassTouch {
    pass: GraphPass,
    access: vk::AccessFlags2KHR,
    stages: vk::PipelineStageFlags2KHR,
}

#[derive(Clone)]
pub(crate) struct ImageSubresource {
    src_image: GraphImage,
    layout: TypeSome<vk::ImageLayout>,
    queue_family: u32,
    access: SmallVec<[PassTouch; 2]>,
}

#[derive(Clone)]
pub(crate) enum ResourceState<T: ResourceMarker> {
    Uninit,
    MoveDst {
        parts: SmallVec<[ImageSubresource; 2]>,
    },
    Normal {
        // TODO perhaps store whether this resource is SHARING_MODE_CONCURRENT
        layout: T::IfImage<vk::ImageLayout>,
        queue_family: u32,
        access: SmallVec<[PassTouch; 2]>,
    },
    Moved,
}

impl<T: ResourceMarker> Default for ResourceState<T> {
    fn default() -> Self {
        Self::Uninit
    }
}

#[derive(Clone)]
pub(crate) struct PassEffects {
    access: vk::AccessFlags2KHR,
    stages: vk::PipelineStageFlags2KHR,
    last_barrier: OptionalU32,
}

#[derive(Clone)]
pub(crate) struct OpenSubmision {
    queue: GraphQueue,
    current_pass: Option<TimelinePass>,
    pass_effects: Vec<PassEffects>,
    passes: Vec<GraphPass>,
    semaphore_dependencies: Vec<GraphSubmission>,
    barriers: Vec<(SubmissionPass, SimpleBarrier)>,
    special_barriers: Vec<(SubmissionPass, SpecialBarrier)>,
}

impl OpenSubmision {
    pub(crate) fn add_pass(
        &mut self,
        timeline: TimelinePass,
        pass: GraphPass,
        effects: PassEffects,
    ) {
        self.current_pass = Some(timeline);
        self.passes.push(pass);
        self.pass_effects.push(effects);
    }
    pub(crate) fn get_current_timeline_pass(&self) -> TimelinePass {
        self.current_pass.unwrap()
    }
    pub(crate) fn get_current_submission_pass(&self) -> SubmissionPass {
        SubmissionPass::new(self.passes.len().checked_sub(1).unwrap())
    }
    pub(crate) fn add_dependency(&mut self, dependency: GraphSubmission) {
        if !self.semaphore_dependencies.contains(&dependency) {
            self.semaphore_dependencies.push(dependency);
        }
    }
    pub(crate) fn finish(&mut self, queue: GraphQueue) -> Submission {
        let take = Submission {
            queue,
            passes: self.passes.clone(),
            semaphore_dependencies: self.semaphore_dependencies.clone(),
            barriers: self.barriers.clone(),
            special_barriers: self.special_barriers.clone(),
            // see the comment in `Submission`
            // the end all barrier is only added after the submission is closed
            contains_end_all_barrier: false,
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

pub(crate) struct SubmissionRecorder<'a> {
    graph: &'a GraphCompiler,
    submissions: Vec<Submission>,
    queues: Vec<OpenSubmision>,
    current_queue: Option<(u32, GraphQueue)>,
}

impl<'a> SubmissionRecorder<'a> {
    pub(crate) fn new(graph: &'a GraphCompiler) -> Self {
        Self {
            graph,
            submissions: Default::default(),
            queues: Default::default(),
            current_queue: Default::default(),
        }
    }
    pub(crate) fn set_current_queue(&mut self, queue: GraphQueue) {
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
    pub(crate) fn get_current_submission(&self) -> &OpenSubmision {
        &self.queues[self.current_queue.unwrap().0 as usize]
    }
    #[inline]
    pub(crate) fn get_current_submission_mut(&mut self) -> &mut OpenSubmision {
        &mut self.queues[self.current_queue.unwrap().0 as usize]
    }
    pub(crate) fn get_current_pass(&self) -> SubmissionPass {
        SubmissionPass::new(
            self.get_current_submission()
                .passes
                .len()
                .checked_sub(1)
                .unwrap(),
        )
    }
    pub(crate) fn get_closed_submission(&self, submission: GraphSubmission) -> &Submission {
        &self.submissions[submission.index()]
    }
    pub(crate) fn get_closed_submission_mut(
        &mut self,
        submission: GraphSubmission,
    ) -> &mut Submission {
        &mut self.submissions[submission.index()]
    }
    pub(crate) fn find_queue_with_family(&self, family: u32) -> Option<GraphQueue> {
        self.queues
            .iter()
            .find(|q| self.graph.input.get_queue_family(q.queue) == family)
            .map(|found| found.queue)
    }
    pub(crate) fn add_submission_sneaky(&mut self, submission: Submission) -> GraphSubmission {
        // TODO is this neccessary?
        assert!(self.current_queue.map(|q| q.1) != Some(submission.queue));
        let index = self.submissions.len();
        self.submissions.push(submission);
        GraphSubmission::new(index)
    }
    pub(crate) fn begin_pass<F: FnOnce(&mut Self)>(
        &mut self,
        timeline: TimelinePass,
        pass: GraphPass,
        on_prev_end: F,
    ) -> SubmissionPass {
        let data = self.graph.input.get_pass_data(pass);

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

        // borrowchk woes
        {
            let submission = self.get_current_submission();
            let meta = self.graph.get_pass_meta(pass);
            meta.scheduled_submission_position
                .set(OptionalU32::new_some(submission.passes.len() as u32))
        }

        let submission = self.get_current_submission_mut();
        submission.add_pass(timeline, pass, effects);

        self.get_current_pass()
    }
    pub(crate) fn add_special_barrier(&mut self, barrier: SpecialBarrier) {
        let pass = self.get_current_pass();
        self.get_current_submission_mut()
            .special_barriers
            .push((pass, barrier));
    }
    pub(crate) fn add_semaphore_dependency(&mut self, dependency: GraphSubmission) {
        self.get_current_submission_mut().add_dependency(dependency);
    }
    pub(crate) fn add_dependency(
        &mut self,
        src_pass: GraphPass,
        src_access: vk::AccessFlags2KHR,
        src_stages: vk::PipelineStageFlags2KHR,
        dst_access: vk::AccessFlags2KHR,
        dst_stages: vk::PipelineStageFlags2KHR,
    ) {
        let (_, queue) = self.current_queue.unwrap();

        let data = self.graph.input.get_pass_data(src_pass);
        let meta = self.graph.get_pass_meta(src_pass);

        let scheduled = meta.scheduled_submission.get();

        // the pass is on another queue in an open submission, close it for the next step
        if data.queue != queue && scheduled.is_none() {
            let _submission = self
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
    pub(crate) fn layout_transition(
        &mut self,
        image: GraphImage,
        src_dst_layout: Range<vk::ImageLayout>,
    ) {
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
    pub(crate) fn close_current_submission(&mut self) {
        self.__close_submission(self.current_queue.unwrap().1);
        self.current_queue = None;
    }
    pub(crate) fn __close_submission(&mut self, queue: GraphQueue) -> Option<GraphSubmission> {
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
    pub(crate) fn finish(mut self) -> Vec<Submission> {
        // cleanup all open submissions, order doesn't matter since they're leaf nodes
        for queue in 0..self.queues.len() {
            self.__close_submission(GraphQueue::new(queue));
        }
        self.submissions
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub(crate) enum ResourceLastUse {
    None,
    Single(GraphSubmission, SmallVec<[SubmissionPass; 4]>),
    Multiple(SmallVec<[GraphSubmission; 4]>),
}

// a thin graph impl for the graph of queue submissions
pub(crate) struct SubmissionFacade<'a>(&'a Vec<Submission>, ());
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
    fn get_node_data(&self, _this: NodeKey) -> &Self::NodeData {
        &()
    }
    fn get_node_data_mut(&mut self, _this: NodeKey) -> &mut Self::NodeData {
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
    pub(crate) fn begin(
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
struct ResourceAccessEntry {
    closed: bool,
    accessors: SmallVec<[TimelinePass; 4]>,
    dst_layout: vk::ImageLayout,
    dst_queue_family: u32,
}

#[derive(Default)]
pub(crate) struct ResourceFirstAccess {
    pub(crate) accessors: SmallVec<[GraphSubmission; 4]>,
    pub(crate) dst_layout: vk::ImageLayout,
    pub(crate) dst_queue_family: u32,
}

pub(crate) enum ImageKindCreateInfo<'a> {
    Image(&'a object::ImageCreateInfo),
    Swapchain(&'a object::SwapchainCreateInfo),
}

pub struct GraphCompiler {
    pub(crate) input: CompilationInput,

    pub(crate) pass_meta: Vec<PassMeta>,
    pub(crate) image_meta: Vec<ImageMeta>,
    pub(crate) buffer_meta: Vec<BufferMeta>,
    alive_passes: Cell<FixedBitSet>,
    alive_images: Cell<FixedBitSet>,
    alive_buffers: Cell<FixedBitSet>,

    pub(crate) pass_children: ImmutableGraph,

    pub(crate) graphics_pipeline_promises: Vec<CompileGraphicsPipelinesTask>,
    pub(crate) function_promises: Vec<ExecuteFnTask>,

    pub(crate) memory: RefCell<ManuallyDrop<Box<[Suballocator; vk::MAX_MEMORY_TYPES as usize]>>>,
    pub(crate) physical_images: Vec<PhysicalImageData>,
    pub(crate) physical_buffers: Vec<PhysicalBufferData>,
}

fn on_cell_mut<T: Default, F: FnOnce(&mut T) -> A, A>(cell: &Cell<T>, fun: F) -> A {
    let mut prev = cell.take();
    let res = fun(&mut prev);
    let _ = cell.replace(prev);
    res
}

impl GraphCompiler {
    pub(crate) fn mark_pass_alive(&self, pass: GraphPass) {
        let data = self.input.get_pass_data(pass);
        let meta = self.get_pass_meta(pass);

        // if true, we have already touched its dependencies and can safely return
        if self.is_pass_alive(pass) {
            return;
        }
        self.make_pass_alive(pass);

        for i in &data.images {
            self.mark_image_alive(i.handle);
        }
        for b in &data.buffers {
            self.make_buffer_alive(b.handle);
        }

        for p in &data.dependencies {
            // only hard dependencies propagate aliveness
            if p.is_hard() {
                self.mark_pass_alive(p.get_pass());
            }
        }
    }
    pub(crate) fn mark_image_alive(&self, image: GraphImage) {
        let mut image = image;
        loop {
            self.make_image_alive(image);
            match self.input.get_image_data(image) {
                ImageData::Moved(to, ..) => {
                    image = to.get().dst;
                }
                _ => break,
            }
        }
    }
    pub(crate) fn get_image_create_info(&self, image: GraphImage) -> ImageKindCreateInfo<'_> {
        let mut data = self.input.get_image_data(image);
        loop {
            match data {
                ImageData::TransientPrototype(info, _) => {
                    return ImageKindCreateInfo::Image(info);
                }
                ImageData::Transient(physical) => {
                    return ImageKindCreateInfo::Image(
                        &self.physical_images[physical.index()].info,
                    );
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
    pub(crate) fn is_image_external<'a>(&'a self, image: GraphImage) -> bool {
        match self.input.get_concrete_image_data(image) {
            ImageData::TransientPrototype(..) => false,
            ImageData::Imported(_) => true,
            ImageData::Swapchain(_) => true,
            ImageData::Transient(..) => unreachable!(),
            ImageData::Moved(..) => unreachable!(),
        }
    }
    pub(crate) fn is_buffer_external(&self, buffer: GraphBuffer) -> bool {
        match self.input.get_buffer_data(buffer) {
            BufferData::TransientPrototype(..) => false,
            BufferData::Transient(..) => unreachable!(),
            BufferData::Imported(_) => true,
        }
    }
    pub(crate) fn is_image_concurrent(&self, image: GraphImage) -> bool {
        self.get_image_meta(image).concurrent
    }
    pub(crate) fn is_buffer_concurrent(&self, buffer: GraphBuffer) -> bool {
        self.get_buffer_meta(buffer).concurrent
    }
    pub(crate) fn is_pass_alive(&self, pass: GraphPass) -> bool {
        on_cell_mut(&self.alive_passes, |c| c.contains(pass.index()))
    }
    pub(crate) fn make_pass_alive(&self, pass: GraphPass) {
        on_cell_mut(&self.alive_passes, |c| c.put(pass.index()));
    }
    pub(crate) fn is_image_alive(&self, image: GraphImage) -> bool {
        on_cell_mut(&self.alive_images, |c| c.contains(image.index()))
    }
    pub(crate) fn make_image_alive(&self, image: GraphImage) {
        on_cell_mut(&self.alive_images, |c| c.put(image.index()));
    }
    pub(crate) fn is_buffer_alive(&self, buffer: GraphBuffer) -> bool {
        on_cell_mut(&self.alive_buffers, |c| c.contains(buffer.index()))
    }
    pub(crate) fn make_buffer_alive(&self, buffer: GraphBuffer) {
        on_cell_mut(&self.alive_buffers, |c| c.put(buffer.index()));
    }
    pub(crate) fn get_suballocator(&self, memory_type: u32) -> RefMut<Suballocator> {
        RefMut::map(self.memory.borrow_mut(), |m| &mut m[memory_type as usize])
    }
    pub(crate) fn get_physical_image_data(&self, image: PhysicalImage) -> &PhysicalImageData {
        &self.physical_images[image.index()]
    }
    pub(crate) fn get_image_meta(&self, image: GraphImage) -> &ImageMeta {
        &self.image_meta[image.index()]
    }
    pub(crate) fn get_physical_buffer_data(&self, buffer: PhysicalBuffer) -> &PhysicalBufferData {
        &self.physical_buffers[buffer.index()]
    }
    pub(crate) fn get_buffer_meta(&self, buffer: GraphBuffer) -> &BufferMeta {
        &self.buffer_meta[buffer.index()]
    }
    pub(crate) fn get_pass_meta(&self, pass: GraphPass) -> &PassMeta {
        &self.pass_meta[pass.0 as usize]
    }
    pub(crate) fn get_start_passes<'a>(&'a self) -> impl Iterator<Item = GraphPass> + 'a {
        self.input
            .passes
            .iter()
            .enumerate()
            .filter(|&(i, pass)| {
                self.is_pass_alive(GraphPass::new(i)) && pass.dependencies.is_empty()
            })
            .map(|(i, _)| GraphPass::new(i))
    }
    pub(crate) fn get_alive_passes<'a>(&'a self) -> impl Iterator<Item = GraphPass> + 'a {
        (0..self.input.passes.len())
            .map(|i| GraphPass::new(i))
            .filter(|&p| self.is_pass_alive(p))
    }
    pub(crate) fn get_children(&self, pass: GraphPass) -> &[GraphPass] {
        let _meta = &self.pass_meta[pass.index()];
        let children = self.pass_children.get_children(pass.0);
        // sound because handles are repr(transparent)
        unsafe {
            std::slice::from_raw_parts::<'_, GraphPass>(
                children.as_ptr() as *const GraphPass,
                children.len(),
            )
        }
    }
    pub(crate) fn compute_graph_layer(&self, pass: GraphPass, graph_layers: &mut [i32]) -> i32 {
        // either it's -1 and is dead or has already been touched and has a positive number
        let layer = graph_layers[pass.index()];
        if layer == -1 {
            return -1;
        }
        if layer > 0 {
            return layer;
        }
        let max = self
            .input
            .get_dependencies(pass)
            .iter()
            .map(|&d| self.compute_graph_layer(d.get_pass(), graph_layers))
            .max()
            .unwrap();
        let current = max + 1;
        graph_layers[pass.index()] = current;
        current
    }
}

fn create_memory_suballocators() -> Box<[Suballocator; vk::MAX_MEMORY_TYPES as usize]> {
    (0..vk::MAX_MEMORY_TYPES)
        .map(|_| Suballocator::new())
        .collect::<Box<[_]>>()
        .try_into()
        .ok()
        .unwrap()
}

impl GraphCompiler {
    pub fn new() -> Self {
        GraphCompiler {
            input: Default::default(),
            pass_meta: Default::default(),
            image_meta: Default::default(),
            buffer_meta: Default::default(),
            pass_children: Default::default(),
            graphics_pipeline_promises: Default::default(),
            function_promises: Default::default(),
            memory: RefCell::new(ManuallyDrop::new(create_memory_suballocators())),
            physical_images: Default::default(),
            physical_buffers: Default::default(),
            alive_passes: Default::default(),
            alive_images: Default::default(),
            alive_buffers: Default::default(),
        }
    }
    fn compilation_prepare(&mut self) {
        self.pass_meta.clear();
        self.image_meta.clear();
        self.buffer_meta.clear();
        self.alive_passes.get_mut().clear();
        self.alive_images.get_mut().clear();
        self.alive_buffers.get_mut().clear();

        self.pass_meta.resize(
            self.input.passes.len(),
            PassMeta {
                scheduled_submission_position: Cell::new(OptionalU32::NONE),
                scheduled_submission: Cell::new(OptionalU32::NONE),
            },
        );
        self.image_meta
            .extend(self.input.images.iter().map(|i| ImageMeta {
                concurrent: i.is_sharing_concurrent(),
            }));
        self.buffer_meta
            .extend(self.input.buffers.iter().map(|i| BufferMeta {
                concurrent: i.is_sharing_concurrent(),
            }));

        self.alive_passes.get_mut().grow(self.input.passes.len());
        self.alive_images.get_mut().grow(self.input.images.len());
        self.alive_buffers.get_mut().grow(self.input.buffers.len());
    }
    pub fn compile<F: FnOnce(&mut GraphBuilder)>(
        &mut self,
        device: OwnedDevice,
        threadpool: &ThreadPool,
        fun: F,
    ) -> CompiledGraph {
        self.input.clear();

        // get the graph from the user, sound because GraphBuilder is repr(transparent)
        let builder = unsafe { std::mem::transmute::<&mut GraphCompiler, &mut GraphBuilder>(self) };
        fun(builder);

        // reset internal structures in preparation for compilation
        self.compilation_prepare();

        let mut graphic = std::mem::take(&mut self.graphics_pipeline_promises);
        let mut funs = std::mem::take(&mut self.function_promises);

        let (sender, receiver) = std::sync::mpsc::channel();
        let owned_device = device.clone();
        threadpool.spawn(move || {
            use rayon::prelude::*;
            let (a, b) = rayon::join(
                move || {
                    Vec::into_par_iter(graphic)
                        .map(|batch| {
                            let mut get_infos = Vec::new();

                            let modes = batch
                                .batch
                                .into_iter()
                                .filter_map(|comp| match comp.mode {
                                    GraphicsPipelineSrc::Compile(mode, mode_hash) => {
                                        get_infos.push(GraphicsPipelineResult::Compile(mode_hash));

                                        Some(mode)
                                    }
                                    GraphicsPipelineSrc::Wait(lock, mode_hash) => {
                                        get_infos
                                            .push(GraphicsPipelineResult::Wait(lock, mode_hash));
                                        None
                                    }
                                    GraphicsPipelineSrc::Ready(handle) => {
                                        get_infos.push(GraphicsPipelineResult::Ready(handle));
                                        None
                                    }
                                })
                                .collect::<SmallVec<[_; 8]>>();

                            let vec = if !modes.is_empty() {
                                let info = unsafe { batch.pipeline_handle.0.get_create_info() };

                                assert!(matches!(info.render_pass, RenderPassMode::Delayed));

                                // todo possibly try to keep this allocator alive longer
                                let bump = Bump::new();
                                let template = unsafe { info.to_vk(&bump, &owned_device) };
                                // we clone the one info multiple times and then patch it with its RenderPassMode
                                let mut infos = vec![template; modes.len()];

                                for (info, mode) in infos.iter_mut().zip(&modes) {
                                    let mut pnext_head: *const std::ffi::c_void = std::ptr::null();
                                    let (render_pass, subpass) =
                                        raw_info_handle_renderpass(&mode, &mut pnext_head, &bump);

                                    assert!(info.p_next.is_null());
                                    info.p_next = pnext_head;
                                    info.render_pass = render_pass;
                                    info.subpass = subpass;
                                }

                                let (vec, result) = unsafe {
                                    owned_device
                                        .device()
                                        .create_graphics_pipelines(
                                            owned_device.pipeline_cache(),
                                            &infos,
                                            owned_device.allocator_callbacks(),
                                        )
                                        .unwrap()
                                };

                                assert_eq!(result, vk::Result::SUCCESS);

                                vec
                            } else {
                                Vec::new()
                            };

                            let mut maybe_lock = None;

                            let mut compiled_pipelines = vec.into_iter().zip(modes);
                            for i in &mut get_infos {
                                match i {
                                    GraphicsPipelineResult::Compile(mode_hash) => {
                                        let (handle, mode) = compiled_pipelines.next().unwrap();
                                        let lock = maybe_lock.get_or_insert_with(|| unsafe {
                                            batch.pipeline_handle.0.lock_storage()
                                        });

                                        unsafe {
                                            batch
                                                .pipeline_handle
                                                .0
                                                .get_object_data()
                                                .mutable
                                                .get_mut(lock)
                                                .add_promised_pipeline(handle, *mode_hash, mode);
                                        }

                                        *i = GraphicsPipelineResult::CompiledFinal(handle);
                                    }
                                    GraphicsPipelineResult::Wait(_, _)
                                    | GraphicsPipelineResult::Ready(_) => {}
                                    GraphicsPipelineResult::CompiledFinal(_) => unreachable!(),
                                }
                            }

                            drop(maybe_lock);

                            assert!(
                                compiled_pipelines.next().is_none(),
                                "Not all compiled pipelines have been handled"
                            );

                            (batch.pipeline_handle, get_infos)
                        })
                        .collect::<Vec<_>>()
                },
                move || {
                    Vec::into_par_iter(funs)
                        .map(|task| Some((task.fun)()))
                        .collect::<Vec<_>>()
                },
            );

            sender.send((a, b)).unwrap();
        });

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
                pub(crate) fn new_normal(accessor: GraphPass, writing: bool) -> Self {
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

            let mut image_rw = vec![ResourceState::default(); self.input.images.len()];
            let mut buffer_rw = vec![ResourceState::default(); self.input.buffers.len()];

            for &e in &self.input.timeline {
                match e.get() {
                    PassEventData::Pass(p) => {
                        let data = &mut self.input.passes[p.index()];

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
                        let ImageMove { from, to } = self.input.get_pass_move(m);
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
        for (i, pass) in self.input.passes.iter().enumerate() {
            // if this pass is already alive, all of its dependendies must have been touched already and we have nothing to do
            if self.is_pass_alive(GraphPass::new(i)) {
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

        struct GraphFacade<'a>(&'a CompilationInput, ());
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
            fn get_node_data(&self, _this: NodeKey) -> &Self::NodeData {
                &()
            }
            fn get_node_data_mut(&mut self, _this: NodeKey) -> &mut Self::NodeData {
                &mut self.1
            }
        }

        // collect the dependees of alive passes (edges going other way than dependencies)
        let mut graph = std::mem::take(&mut self.pass_children);
        let facade = GraphFacade(&self.input, ());
        reverse_edges_into(&facade, &mut graph);
        let _ = std::mem::replace(&mut self.pass_children, graph);

        // do a greedy graph traversal where nodes with a larger priority will be selected first
        // at this point this is essentially just a bfs
        let scheduled = {
            #[derive(Clone)]
            struct AvailablePass {
                pass: GraphPass,
                priority: i32,
            }

            impl AvailablePass {
                pub(crate) fn new(
                    pass: GraphPass,
                    graph: &GraphCompiler,
                    graph_layers: &mut [i32],
                ) -> Self {
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
                .input
                .passes
                .iter()
                .map(|p| (p.dependencies.len() as u16, false))
                .collect::<Vec<_>>();

            let mut available: Vec<(u32, BinaryHeap<AvailablePass>)> =
                vec![(0, BinaryHeap::new()); self.input.queues.len()];
            let mut scheduled: Vec<GraphPassEvent> = Vec::new();

            let mut graph_layers = vec![0; self.input.passes.len()];
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

            let mut queue_flush_region = vec![0usize; self.input.queues.len()];

            // currently we are creating the scheduled passes by looping over each queue and poppping the locally optimal pass
            // this is rather questionable so TODO think this over
            loop {
                let len = scheduled.len();
                for queue_i in 0..self.input.queues.len() {
                    let (position, heap) = &mut available[queue_i];

                    let pass;
                    if let Some(AvailablePass { pass: p, .. }) = heap.pop() {
                        pass = p;
                    } else {
                        // we've depleted the last flush region, continue to the next one
                        let index = &mut queue_flush_region[queue_i];
                        for &e in &self.input.timeline[*index..] {
                            *index += 1;
                            match e.get() {
                                PassEventData::Pass(next_pass) => {
                                    let data = &self.input.get_pass_data(next_pass);
                                    if data.queue.index() != queue_i {
                                        continue;
                                    }
                                    let dependency_info = &mut dependency_count[next_pass.index()];
                                    // if the pass has no outstanding deendencies, we measure its priority and add it to the heap
                                    if dependency_info.0 == 0 {
                                        let _queue = data.queue;
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
                            let queue = self.input.get_pass_data(child).queue;
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
        let (image_usage, _buffer_usage) = {
            let mut image_usage = vec![0..0; self.input.images.len()];
            let mut buffer_usage = vec![0..0; self.input.buffers.len()];

            for (event_i, e) in scheduled.iter().enumerate() {
                match e.get() {
                    PassEventData::Pass(p) => {
                        let data = self.input.get_pass_data(p);
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
            .input
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
        let (mut submissions, image_last_state, buffer_last_state, accessors) = {
            let mut image_rw: Vec<ResourceState<ImageMarker>> =
                vec![ResourceState::Uninit; self.input.images.len()];
            let mut buffer_rw: Vec<ResourceState<BufferMarker>> =
                vec![ResourceState::Uninit; self.input.buffers.len()];

            // for i in 0..self.input.images.len() {
            //     let image = GraphImage::new(i);
            //     image_rw[i].1 = self.input.get_image_data(image).is_sharing_concurrent();
            // }

            // for i in 0..self.input.buffers.len() {
            //     let buffer = GraphBuffer::new(i);
            //     buffer_rw[i].1 = self.input.get_buffer_data(buffer).is_sharing_concurrent();
            // }

            let recorder = RefCell::new(SubmissionRecorder::new(self));

            // set of first accesses that occur in the timeline,
            // becomes "closed" (bool = true) when the first write occurs
            // this will be used during graph execution to synchronize with external resources
            let mut accessors: HashMap<CombinedResourceHandle, ResourceAccessEntry> =
                constant_ahash_hashmap();

            // // we need to prefill the hashmap?
            // accessors.extend(
            //     self.input
            //         .images
            //         .iter()
            //         .enumerate()
            //         .filter_map(|(i, data)| {
            //             let image = GraphImage::new(i);
            //             let handle = CombinedResourceHandle::new_image(image);
            //             self.is_image_external(image)
            //                 .then_some((handle, ResourceAccessEntry::default()))
            //         }),
            // );

            // accessors.extend(
            //     self.input
            //         .buffers
            //         .iter()
            //         .enumerate()
            //         .filter_map(|(i, data)| {
            //             let buffer = GraphBuffer::new(i);
            //             let handle = CombinedResourceHandle::new_buffer(buffer);
            //             self.is_buffer_external(buffer)
            //                 .then_some((handle, ResourceAccessEntry::default()))
            //         }),
            // );

            for (timeline_i, &e) in scheduled.iter().enumerate() {
                match e.get() {
                    PassEventData::Pass(p) => {
                        let timeline_pass = e.to_timeline_pass().unwrap();
                        let submission_pass =
                            recorder.borrow_mut().begin_pass(timeline_pass, p, |_| {});

                        let data = self.input.get_pass_data(p);
                        let queue = data.queue;
                        let queue_family = self.input.get_queue_family(queue);

                        for img in &data.images {
                            self.handle_resource(
                                p,
                                timeline_pass,
                                submission_pass,
                                queue_family,
                                img,
                                &recorder,
                                &mut image_rw,
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
                            let data = &self.input.moves[mov.index()];

                            let mut new_parts = SmallVec::new();

                            for &src_image in &data.from {
                                match std::mem::replace(
                                    &mut image_rw[src_image.index()],
                                    ResourceState::Moved,
                                ) {
                                    ResourceState::MoveDst { parts } => new_parts.extend(parts),
                                    ResourceState::Normal {
                                        layout,
                                        queue_family,
                                        access,
                                    } => {
                                        new_parts.push(ImageSubresource {
                                            src_image,
                                            layout: layout,
                                            queue_family,
                                            access,
                                        });
                                    }
                                    _ => panic!("Resource in unsupported state"),
                                }
                            }
                            let state = &mut image_rw[data.to.index()];
                            let ResourceState::Uninit = *state else {
                                panic!("Image move destination must be unitialized!");
                            };
                            *state = ResourceState::<ImageMarker>::MoveDst { parts: new_parts };
                        }
                    }
                    PassEventData::Move(_) => {} // this is handled differently
                    PassEventData::Flush(_q) => recorder.borrow_mut().close_current_submission(),
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
                                CombinedResourceHandle::new_image(image)
                            }
                            &SpecialBarrier::ImageOwnershipTransition { image, .. } => {
                                CombinedResourceHandle::new_image(image)
                            }
                            &SpecialBarrier::BufferOwnershipTransition { buffer, .. } => {
                                CombinedResourceHandle::new_buffer(buffer)
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

            let mut submissions_scratch = Vec::new();

            let mut final_accessors: HashMap<CombinedResourceHandle, ResourceFirstAccess> =
                constant_ahash_hashmap();

            final_accessors.extend(accessors.into_iter().map(|(key, accessor)| {
                submissions_scratch.clear();
                submissions_scratch.extend(accessor.accessors.iter().map(|a| {
                    let pass = self.input.timeline[a.index()].get_pass().unwrap();
                    GraphSubmission(self.get_pass_meta(pass).scheduled_submission.get().unwrap())
                }));
                submissions_scratch.sort();
                submissions_scratch.dedup();

                if submissions_scratch.len() == 0 {
                    panic!("Nothing is using the resource? Is this even possible? Wouldn't the resource be dead?");
                }

                let access = ResourceFirstAccess {
                    accessors: SmallVec::from_slice(&submissions_scratch),
                    dst_layout: accessor.dst_layout,
                    dst_queue_family: accessor.dst_queue_family,
                };

                (key, access)
            }));

            (submissions, image_rw, buffer_rw, final_accessors)
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

            let mut submission_reuse = vec![SubmissionResourceReuse::default(); submissions.len()];
            // TODO profile how many children submissions usually have
            // a vector may be faster here
            let mut intersection_hashmap: ahash::HashMap<GraphSubmission, (GraphSubmission, u32)> =
                constant_ahash_hashmap();
            let mut availibility_interner: (
                AvailabilityToken,
                ahash::HashMap<ResourceLastUse, AvailabilityToken>,
            ) = (AvailabilityToken::new(), constant_ahash_hashmap());

            let mut image_available = vec![AvailabilityToken::NONE; self.input.images.len()];
            let image_last_touch = self.get_last_resource_usage(&image_last_state, &mut scratch);
            submission_fill_reuse::<ImageMarker>(
                image_last_touch,
                &mut image_available,
                &mut submission_reuse,
                &mut intersection_hashmap,
                &submission_children,
                &mut availibility_interner,
            );
            let mut buffer_available = vec![AvailabilityToken::NONE; self.input.buffers.len()];
            let buffer_last_touch = self.get_last_resource_usage(&buffer_last_state, &mut scratch);
            submission_fill_reuse::<BufferMarker>(
                buffer_last_touch,
                &mut buffer_available,
                &mut submission_reuse,
                &mut intersection_hashmap,
                &submission_children,
                &mut availibility_interner,
            );

            for allocator in self.memory.borrow_mut().iter_mut() {
                allocator.reset();
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
                    let _a = 2;
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
                    let data = self.input.get_pass_data(p);

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

                    let device_wrapper = device.device();

                    need_alloc.clear();
                    for a in &data.images {
                        if let ImageData::TransientPrototype(create_info, _allocation_info) =
                            self.input.get_image_data(a.handle)
                        {
                            let vk_info = create_info.to_vk();
                            let image = unsafe {
                                device_wrapper
                                    .create_image(&vk_info, device.allocator_callbacks())
                                    .unwrap()
                            };
                            let requirements =
                                unsafe { device_wrapper.get_image_memory_requirements(image) };
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
                        if let BufferData::TransientPrototype(create_info, _allocation_info) =
                            self.input.get_buffer_data(a.handle)
                        {
                            let vk_info = create_info.to_vk();
                            let buffer = unsafe {
                                device_wrapper
                                    .create_buffer(&vk_info, device.allocator_callbacks())
                                    .unwrap()
                            };
                            let requirements =
                                unsafe { device_wrapper.get_buffer_memory_requirements(buffer) };
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
                            NeedAlloc::Image {
                                vkhandle: _,
                                handle,
                            } => {
                                let ImageData::TransientPrototype(create_info, allocation_info) = self.input.get_image_data(handle) else {
                                    unreachable!()
                                };
                                (
                                    create_info.tiling.into(),
                                    allocation_info,
                                    image_available[handle.index()],
                                    self.input.get_image_display(handle),
                                )
                            }
                            NeedAlloc::Buffer {
                                vkhandle: _,
                                handle,
                            } => {
                                let BufferData::TransientPrototype(_create_info, allocation_info) = self.input.get_buffer_data(handle) else {
                                    unreachable!()
                                };
                                (
                                    MemoryKind::Linear,
                                    allocation_info,
                                    buffer_available[handle.index()],
                                    self.input.get_buffer_display(handle),
                                )
                            }
                        };

                        let allocator = device.allocator();
                        let buffer_image_granularity = device
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
                            unsafe { device.allocator().get_allocation_info(allocation) };

                        match *resource {
                            NeedAlloc::Image { vkhandle, handle } => {
                                let data = self.input.get_image_data(handle);
                                let ImageData::TransientPrototype(create_info, _allocation_info) = data else {
                                    unreachable!()
                                };
                                let _allocation = suballocation.memory.allocation;

                                unsafe {
                                    // TODO consider using bind_image_memory2
                                    device
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
                                    state: RefCell::new(ImageMutableState::new(
                                        vk::ImageLayout::UNDEFINED,
                                    )),
                                });
                                *self.input.get_image_data_mut(handle) =
                                    ImageData::Transient(physical);
                            }
                            NeedAlloc::Buffer { vkhandle, handle } => {
                                let data = self.input.get_buffer_data(handle);
                                let BufferData::TransientPrototype(create_info, _allocation_info) = data else {
                                    unreachable!()
                                };
                                let _allocation = suballocation.memory.allocation;

                                unsafe {
                                    // TODO consider using bind_buffer_memory2
                                    device
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
                                    state: RefCell::new(BufferMutableState::new()),
                                });
                                *self.input.get_buffer_data_mut(handle) =
                                    BufferData::Transient(physical);
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
        }

        let (graphics_pipelines, function_promises): (
            Vec<(GraphicsPipeline, Vec<GraphicsPipelineResult>)>,
            Vec<Option<SendAny>>,
        ) = receiver.recv().unwrap();

        let mut graph_context = GraphContext {
            graphics_pipelines,
            function_promises,
        };

        for i in 0..self.input.passes.len() {
            if self.is_pass_alive(GraphPass::new(i)) {
                self.input.passes[i].create_pass(&mut graph_context);
            }
        }

        for (key, access) in &accessors {
            let &ResourceFirstAccess {
                ref accessors,
                dst_layout,
                dst_queue_family,
            } = access;

            match key.unpack() {
                GraphResource::Image(image) => {
                    let data = self.input.get_image_data(image);
                    let meta = self.get_image_meta(image);

                    match data {
                        ImageData::Swapchain(_) => {
                            assert!(accessors.len() == 1, "Swapchains use legacy semaphores and do not support multiple signals or waits, using a swapchain in multiple submissions is disallowed (you should really just transfer the final image into it at the end of the frame)");

                            let first_access_submission = {
                                let a = accessors[0];
                                let pass = self.input.timeline[a.index()].get_pass().unwrap();
                                let submission =
                                    self.get_pass_meta(pass).scheduled_submission.get().unwrap();
                                &mut submissions[submission as usize]
                            };

                            if dst_layout != vk::ImageLayout::UNDEFINED {
                                // acquired images start out as UNDEFINED, since we are currently allowing swapchains to only be used in a single submission,
                                // we can just transition the layout at the start of the submission
                                first_access_submission.special_barriers.insert(
                                    0,
                                    (
                                        SubmissionPass(0),
                                        SpecialBarrier::LayoutTransition {
                                            image,
                                            src_layout: vk::ImageLayout::UNDEFINED,
                                            dst_layout,
                                        },
                                    ),
                                )
                            }

                            let src_layout = match &image_last_state[image.index()] {
                                ResourceState::Uninit => {
                                    panic!("Must write to swapchain before presenting")
                                }
                                ResourceState::Normal { layout, .. } => layout.unwrap(),
                                ResourceState::MoveDst { .. } | ResourceState::Moved => {
                                    panic!("Swapchains do not support moving")
                                }
                            };

                            // swapchain images must be in PRESENT_SRC_KHR before presenting
                            // FIXME if we support moving into external resources (which we should) this will have to be intergrated
                            // with the full barrier emission logic because this is not enough for MoveDst states
                            first_access_submission.add_final_special_barrier(
                                SpecialBarrier::LayoutTransition {
                                    image,
                                    src_layout: src_layout,
                                    dst_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                                },
                            )
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        CompiledGraph {
            input: std::mem::take(&mut self.input),
            timeline: std::mem::take(&mut self.input.timeline),
            submissions,
            external_resource_initial_access: accessors,
            physical_images: std::mem::take(&mut self.physical_images),
            physical_buffers: std::mem::take(&mut self.physical_buffers),
            memory: RefCell::into_inner(std::mem::replace(
                &mut self.memory,
                RefCell::new(ManuallyDrop::new(create_memory_suballocators())),
            )),
            image_last_state,
            buffer_last_state,
            alive_passes: self.alive_passes.take(),
            alive_images: self.alive_images.take(),
            alive_buffers: self.alive_buffers.take(),

            state: CompiledGraphState::new(device),
        }
    }

    fn handle_resource<T: ResourceMarker>(
        &self,
        p: GraphPass,
        timeline_pass: TimelinePass,
        submission_pass: SubmissionPass,
        queue_family: u32,
        img: &T::Data,
        recorder: &RefCell<SubmissionRecorder>,
        image_rw: &mut Vec<ResourceState<T>>,
        accessors: &mut ahash::HashMap<CombinedResourceHandle, ResourceAccessEntry>,
    ) where
        // hack because without this the typechecker is not cooperating
        T::IfImage<vk::ImageLayout>: Copy,
    {
        let writes =
            self.emit_barriers::<T>(p, submission_pass, queue_family, img, recorder, image_rw);

        let resource = img.graph_resource();
        let imported = match resource.unpack() {
            GraphResource::Image(handle) => {
                matches!(
                    self.input.get_image_data(handle),
                    ImageData::Imported(_) | ImageData::Swapchain(_)
                )
            }
            GraphResource::Buffer(handle) => {
                matches!(self.input.get_buffer_data(handle), BufferData::Imported(_))
            }
        };

        if imported {
            let ResourceAccessEntry {
                closed, accessors, ..
            } = accessors.entry(resource).or_insert_with(|| {
                let ResourceState::Normal {
                    layout,
                    queue_family,
                    access: _,
                } = &image_rw[img.raw_resource_handle().index()] else {
                    panic!("Impossible for external resources to be any other state")
                };

                ResourceAccessEntry {
                    closed: false,
                    accessors: smallvec![],
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

    pub(crate) fn get_last_resource_usage<'a, T: ResourceMarker>(
        &'a self,
        resource_last_state: &'a [ResourceState<T>],
        scratch: &'a mut Vec<(GraphSubmission, SubmissionPass)>,
    ) -> impl Iterator<Item = ResourceLastUse> + 'a {
        resource_last_state.iter().map(|state| {
            pub(crate) fn add(
                from: impl Iterator<Item = PassTouch> + Clone,
                to: &mut Vec<(GraphSubmission, SubmissionPass)>,
                graph: &GraphCompiler,
            ) {
                let parts = from.map(|touch| {
                    let meta = graph.get_pass_meta(touch.pass);
                    let submission = GraphSubmission(meta.scheduled_submission.get().unwrap());
                    let pass = SubmissionPass(meta.scheduled_submission_position.get().unwrap());
                    (submission, pass)
                });
                to.extend(parts);
            }

            scratch.clear();
            match state {
                ResourceState::Uninit | ResourceState::Moved => {
                    return ResourceLastUse::None;
                }
                ResourceState::MoveDst { parts } => {
                    add(
                        parts
                            .iter()
                            .flat_map(|subresource| &subresource.access)
                            .cloned(),
                        scratch,
                        self,
                    );
                }
                ResourceState::Normal {
                    layout: _,
                    queue_family: _,
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

    pub(crate) fn emit_barriers<T: ResourceMarker>(
        &self,
        pass: GraphPass,
        _dst_pass: SubmissionPass,
        dst_queue_family: u32,
        resource_data: &T::Data,
        recorder: &RefCell<SubmissionRecorder>,
        resource_rw: &mut [ResourceState<T>],
    ) -> bool
    where
        // hack because without this the typechecker is not cooperating
        T::IfImage<vk::ImageLayout>: Copy,
    {
        let raw_resource_handle = resource_data.raw_resource_handle();
        let resource_handle = resource_data.graph_resource();
        let state = &mut resource_rw[raw_resource_handle.index()];
        let concurrent = match resource_handle.unpack() {
            GraphResource::Image(image) => self.is_image_concurrent(image),
            GraphResource::Buffer(buffer) => self.is_buffer_concurrent(buffer),
        };
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
                let imported = match resource_handle.unpack() {
                    GraphResource::Image(handle) => {
                        matches!(
                            self.input.get_image_data(handle),
                            ImageData::Imported(_) | ImageData::Swapchain(_)
                        )
                    }
                    GraphResource::Buffer(handle) => {
                        matches!(self.input.get_buffer_data(handle), BufferData::Imported(_))
                    }
                };
                // layout transition
                if T::IS_IMAGE && !imported && dst_layout.unwrap() != vk::ImageLayout::UNDEFINED {
                    recorder.borrow_mut().layout_transition(
                        GraphImage(raw_resource_handle.0),
                        vk::ImageLayout::UNDEFINED..dst_layout.unwrap(),
                    );
                }

                *state = normal_state();
            }
            // if the current access only needs to read, add it to the readers (and handle layout transitions)
            // otherwise synchronize against all passes and transition to normal state
            ResourceState::MoveDst { parts } => {
                assert!(ImageMarker::IS_IMAGE, "Only images can be moved");
                let dst_layout = TypeSome::new_some(dst_layout.unwrap());
                let mut all_parts_written_to = true;

                for subresource in parts.iter_mut() {
                    let ImageSubresource {
                        src_image: _,
                        layout: src_layout,
                        queue_family: src_queue_family,
                        access,
                    } = subresource;

                    let mut dst_writes_copy = dst_writes;

                    self.handle_resource_state_normal::<ImageMarker>(
                        &access,
                        dst_touch,
                        src_queue_family,
                        dst_queue_family,
                        src_layout,
                        dst_layout,
                        &mut dst_writes_copy,
                        raw_resource_handle,
                        resource_handle,
                        concurrent,
                        recorder,
                    );

                    if dst_writes {
                        access.clear();
                    } else {
                        all_parts_written_to = false;
                    }

                    access.push(dst_touch);
                }

                if all_parts_written_to {
                    *state = normal_state();
                }
            }
            ResourceState::Normal {
                layout: src_layout,
                queue_family: src_queue_family,
                access,
            } => {
                self.handle_resource_state_normal::<T>(
                    access,
                    dst_touch,
                    src_queue_family,
                    dst_queue_family,
                    src_layout,
                    dst_layout,
                    &mut dst_writes,
                    raw_resource_handle,
                    resource_handle,
                    concurrent,
                    recorder,
                );

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
    pub(crate) fn handle_resource_state_normal<T: ResourceMarker>(
        &self,
        access: &[PassTouch],
        dst_touch: PassTouch,
        src_queue_family: &mut u32,
        dst_queue_family: u32,
        src_layout: &mut <T as ResourceMarker>::IfImage<vk::ImageLayout>,
        dst_layout: <T as ResourceMarker>::IfImage<vk::ImageLayout>,
        dst_writes: &mut bool,
        raw_resource_handle: RawHandle,
        resource_handle: CombinedResourceHandle,
        concurrent: bool,
        recorder: &RefCell<SubmissionRecorder>,
    ) where
        <T as ResourceMarker>::IfImage<vk::ImageLayout>: Copy,
    {
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

        if *dst_writes {
            synchronize();
        }

        // layout transition
        if T::IS_IMAGE
            && dst_layout.unwrap() != vk::ImageLayout::UNDEFINED
            && dst_layout.unwrap() != src_layout.unwrap()
        {
            synchronize();
            *dst_writes = true;
            recorder.borrow_mut().layout_transition(
                GraphImage(raw_resource_handle.0),
                src_layout.unwrap()..dst_layout.unwrap(),
            );
        }
        // We want to set the layout to UNDEFINED even if no layout transition happened because the contents cannot be trusted afterwards
        *src_layout = dst_layout;

        // queue family ownership transition
        if !concurrent
            // if we do not need the contents of the resource (=layout UNDEFINED), we can simply ignore the transfer
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-queue-transfers
            && (T::IS_BUFFER || dst_layout.unwrap() != vk::ImageLayout::UNDEFINED)
            && *src_queue_family != dst_queue_family
        {
            synchronize();
            *dst_writes = true;

            self.emit_family_ownership_transition(
                resource_handle,
                *src_queue_family,
                dst_queue_family,
                access,
                &mut recorder.borrow_mut(),
            );
        }
    }
    pub(crate) fn emit_family_ownership_transition(
        &self,
        resource_handle: CombinedResourceHandle,
        src_queue_family: u32,
        dst_queue_family: u32,
        access: &[PassTouch],
        recorder: &mut SubmissionRecorder,
    ) {
        let get_submission_for_pass = |pass: GraphPass| -> (GraphSubmission, SubmissionPass) {
            let src_meta = self.get_pass_meta(pass);
            let submission = GraphSubmission(src_meta.scheduled_submission.get().unwrap());
            let pass = SubmissionPass(src_meta.scheduled_submission_position.get().unwrap());

            (submission, pass)
        };

        let barrier = match resource_handle.unpack() {
            GraphResource::Image(image) => SpecialBarrier::ImageOwnershipTransition {
                image,
                src_family: src_queue_family,
                dst_family: dst_queue_family,
            },
            GraphResource::Buffer(buffer) => SpecialBarrier::BufferOwnershipTransition {
                buffer,
                src_family: src_queue_family,
                dst_family: dst_queue_family,
            },
        };

        assert!(!access.is_empty(), "For the resource to get into a Normal state it must have been first accessed so it must not have the access field empty");

        let mut access_submissions = access
            .iter()
            .map(|t| get_submission_for_pass(t.pass))
            .collect::<SmallVec<[_; 16]>>();

        // if all of the accesses are within the same submission, we can just add a dummy pass at the end
        // which waits for all of the passes and then releases ownership
        if access_submissions[1..]
            .iter()
            .all(|&(submission, _)| submission == access_submissions[0].0)
        {
            let sub = recorder.get_closed_submission_mut(access_submissions[0].0);
            sub.add_final_special_barrier(barrier);
        } else {
            // there are multiple submissions which we need to synchronize against to transfer the ownership
            // the only way to wait for them on a src_queue family is to create a dummy submission which binds them together
            // TODO merge dummy submissions where possible

            access_submissions.sort_by_key(|&(submission, _)| submission);
            access_submissions.dedup_by_key(|&mut (submission, _)| submission);

            let queue = recorder.find_queue_with_family(src_queue_family).unwrap();

            let submission = Submission {
                queue,
                passes: Default::default(),
                semaphore_dependencies: access_submissions.iter().map(|&(s, _)| s).collect(),
                barriers: Default::default(),
                special_barriers: vec![(
                    Submission::AFTER_END_ALL_BARRIER_SUBMISSION_PASS,
                    barrier,
                )],
                contains_end_all_barrier: false,
            };

            let sub = recorder.add_submission_sneaky(submission);
            recorder.add_semaphore_dependency(sub);
        }
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
        let _handle = RawHandle::new(i);
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

pub(crate) struct SwapchainPresent {
    pub(crate) vkhandle: vk::SwapchainKHR,
    pub(crate) image_index: u32,
    pub(crate) image_acquire: vk::Semaphore,
    pub(crate) image_release: vk::Semaphore,
}
