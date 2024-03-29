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
        debug::maybe_attach_debug_label,
        submission::{self, QueueSubmission},
        Device, OwnedDevice,
    },
    graph::{
        allocator::MemoryKind,
        execute::{CompiledGraphVulkanState, GraphExecutor, MainCompiledGraphVulkanState},
        resource_marker::{BufferMarker, ImageMarker, TypeOption},
        reverse_edges::reverse_edges_into,
        task::{GraphicsPipelineSrc, SendUnsafeCell},
    },
    object::{
        self, raw_info_handle_renderpass, BufferMutableState, ConcreteGraphicsPipeline,
        GetPipelineResult, GraphicsPipeline, ImageMutableState, RenderPassMode,
        SwapchainAcquireStatus, SynchronizeResult,
    },
    passes::RenderPass,
    simple_handle,
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
    ops::{Deref, DerefMut, Range},
    sync::Arc,
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

impl PassObjectState {
    // pub(crate) fn on_pass<A, F: FnOnce(&mut PassObjectState) -> A>(&mut self, fun: F) -> A {
    //     // Cell::replace just moves a pointer, easy for compiler to optimize
    //     let mut pass = Cell::replace(self, PassObjectState::DummyNone);
    //     let ret = fun(&mut pass);
    //     let _ = Cell::replace(self, pass);
    //     ret
    // }
    fn on_initial<A, F: FnOnce(&mut dyn ObjectSafeCreatePass) -> A>(&mut self, fun: F) -> A {
        match self {
            PassObjectState::Initial(initial) => fun(&mut **initial),
            PassObjectState::Created(_) => panic!(),
            PassObjectState::DummyNone => panic!(),
        }
    }
    pub(crate) fn create_pass(&mut self, ctx: &mut GraphContext) {
        match self {
            PassObjectState::Initial(initial) => {
                let created = initial.create(ctx);
                *self = PassObjectState::Created(created);
            }
            PassObjectState::Created(_) => panic!(),
            PassObjectState::DummyNone => panic!(),
        }
    }
    pub(crate) fn on_created<A, F: FnOnce(&mut dyn RenderPass) -> A>(&mut self, fun: F) -> A {
        match self {
            PassObjectState::Initial(_) => panic!(),
            PassObjectState::Created(created) => fun(&mut **created),
            PassObjectState::DummyNone => panic!(),
        }
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
                    archandle.0.access_mutable(
                        |d| &d.mutable,
                        |m| m.get_pipeline(*mode_hash, || unreachable!()),
                    )
                };

                unsafe { lock.unlock_shared() };

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
pub struct CombinedResourceHandle(PackedUint<CombinedResourceConfig, u32>);

impl CombinedResourceHandle {
    pub(crate) fn new_image(image: GraphImage) -> Self {
        Self(PackedUint::new(0, image.0))
    }
    pub(crate) fn new_buffer(buffer: GraphBuffer) -> Self {
        Self(PackedUint::new(1, buffer.0))
    }
    pub(crate) fn get_image(self) -> Option<GraphImage> {
        if self.0.first() == 0 {
            Some(GraphImage(self.0.second()))
        } else {
            None
        }
    }
    pub(crate) fn get_buffer(self) -> Option<GraphBuffer> {
        if self.0.first() == 1 {
            Some(GraphBuffer(self.0.second()))
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CombinedBarrierConfig;
impl Config for CombinedBarrierConfig {
    const FIRST_BITS: usize = 2;
    const SECOND_BITS: usize = 30;
}

simple_handle! {pub(crate) PassMemoryBarrier, pub(crate) PassImageBarrier, pub(crate) PassBufferBarrier}

impl PassMemoryBarrier {
    fn to_combined(self) -> CombinedBarrierHandle {
        CombinedBarrierHandle::new_memory(self)
    }
}
impl PassImageBarrier {
    fn to_combined(self) -> CombinedBarrierHandle {
        CombinedBarrierHandle::new_image(self)
    }
}
impl PassBufferBarrier {
    fn to_combined(self) -> CombinedBarrierHandle {
        CombinedBarrierHandle::new_buffer(self)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum SubmissionBarrier {
    Memory(PassMemoryBarrier),
    Image(PassImageBarrier),
    Buffer(PassBufferBarrier),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CombinedBarrierHandle(PackedUint<CombinedBarrierConfig, u32>);

impl CombinedBarrierHandle {
    pub(crate) fn new_memory(memory: PassMemoryBarrier) -> Self {
        Self(PackedUint::new(0, memory.0))
    }
    pub(crate) fn new_image(image: PassImageBarrier) -> Self {
        Self(PackedUint::new(1, image.0))
    }
    pub(crate) fn new_buffer(buffer: PassBufferBarrier) -> Self {
        Self(PackedUint::new(2, buffer.0))
    }
    pub(crate) fn unpack(self) -> SubmissionBarrier {
        let second = self.0.second();
        match self.0.first() {
            0 => SubmissionBarrier::Memory(PassMemoryBarrier(second)),
            1 => SubmissionBarrier::Image(PassImageBarrier(second)),
            2 => SubmissionBarrier::Buffer(PassBufferBarrier(second)),
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
pub(crate) struct ResourceSubresource<T: ResourceMarker>
where
    T::IfImage<vk::ImageLayout>: Clone,
{
    pub(crate) read_barrier: Option<CombinedBarrierHandle>,
    pub(crate) last_write: Option<PassTouch>,
    pub(crate) layout: T::IfImage<vk::ImageLayout>,
    pub(crate) queue_family: u32,
    pub(crate) access: SmallVec<[PassTouch; 2]>,
}

#[derive(Clone)]
pub(crate) enum ResourceState<T: ResourceMarker>
where
    T::IfImage<vk::ImageLayout>: Clone,
{
    Uninit,
    MoveDst {
        parts: SmallVec<[(GraphImage, ResourceSubresource<T>); 2]>,
    },
    Normal(ResourceSubresource<T>),
    Moved,
}

impl<T: ResourceMarker> Default for ResourceState<T>
where
    T::IfImage<vk::ImageLayout>: Clone,
{
    fn default() -> Self {
        Self::Uninit
    }
}

#[derive(Clone)]
pub(crate) struct MemoryBarrier {
    pub(crate) src_stages: vk::PipelineStageFlags2KHR,
    pub(crate) dst_stages: vk::PipelineStageFlags2KHR,
    pub(crate) src_access: vk::AccessFlags2KHR,
    pub(crate) dst_access: vk::AccessFlags2KHR,
}

impl MemoryBarrier {
    pub(crate) fn to_vk(&self) -> vk::MemoryBarrier2KHR {
        vk::MemoryBarrier2KHR {
            src_stage_mask: self.src_stages,
            src_access_mask: self.src_access,
            dst_stage_mask: self.dst_stages,
            dst_access_mask: self.dst_access,
            ..Default::default()
        }
    }
}

#[derive(Clone)]
pub(crate) struct ImageBarrier {
    pub(crate) image: GraphImage,
    pub(crate) src_stages: vk::PipelineStageFlags2KHR,
    pub(crate) dst_stages: vk::PipelineStageFlags2KHR,
    pub(crate) src_access: vk::AccessFlags2KHR,
    pub(crate) dst_access: vk::AccessFlags2KHR,
    pub(crate) old_layout: vk::ImageLayout,
    pub(crate) new_layout: vk::ImageLayout,
    pub(crate) src_queue_family_index: u32,
    pub(crate) dst_queue_family_index: u32,
}

impl ImageBarrier {
    pub(crate) fn to_vk(
        &self,
        image: vk::Image,
        subresource_range: vk::ImageSubresourceRange,
    ) -> vk::ImageMemoryBarrier2KHR {
        vk::ImageMemoryBarrier2KHR {
            image,
            src_stage_mask: self.src_stages,
            src_access_mask: self.src_access,
            dst_stage_mask: self.dst_stages,
            dst_access_mask: self.dst_access,
            old_layout: self.old_layout,
            new_layout: self.new_layout,
            src_queue_family_index: self.src_queue_family_index,
            dst_queue_family_index: self.dst_queue_family_index,
            subresource_range,
            ..Default::default()
        }
    }
}

#[derive(Clone)]
pub(crate) struct BufferBarrier {
    pub(crate) buffer: GraphBuffer,
    pub(crate) src_stages: vk::PipelineStageFlags2KHR,
    pub(crate) dst_stages: vk::PipelineStageFlags2KHR,
    pub(crate) src_access: vk::AccessFlags2KHR,
    pub(crate) dst_access: vk::AccessFlags2KHR,
    pub(crate) src_queue_family_index: u32,
    pub(crate) dst_queue_family_index: u32,
}

impl BufferBarrier {
    pub(crate) fn to_vk(&self, buffer: vk::Buffer) -> vk::BufferMemoryBarrier2KHR {
        vk::BufferMemoryBarrier2KHR {
            buffer,
            src_stage_mask: self.src_stages,
            src_access_mask: self.src_access,
            dst_stage_mask: self.dst_stages,
            dst_access_mask: self.dst_access,
            src_queue_family_index: self.src_queue_family_index,
            dst_queue_family_index: self.dst_queue_family_index,
            offset: 0,
            size: vk::WHOLE_SIZE,
            ..Default::default()
        }
    }
}

#[derive(Clone)]
pub(crate) struct Submission {
    pub(crate) queue: GraphQueue,
    pub(crate) passes: Vec<GraphPass>,
    pub(crate) semaphore_dependencies: Vec<GraphSubmission>,
    // barriers may have SubmissionPass targets that go beyond the actual passes vector
    // these are emitted to fixup synchronization for queue ownership transfer
    pub(crate) memory_barriers: Vec<(SubmissionPass, MemoryBarrier)>,
    pub(crate) image_barriers: Vec<(SubmissionPass, ImageBarrier)>,
    pub(crate) buffer_barriers: Vec<(SubmissionPass, BufferBarrier)>,
}

impl Submission {
    const FINAL_SUBMISSION_BARRIER_PASS: SubmissionPass = SubmissionPass(u32::MAX);
    fn new(queue: GraphQueue) -> Self {
        Self {
            queue,
            passes: Default::default(),
            semaphore_dependencies: Default::default(),
            memory_barriers: Default::default(),
            image_barriers: Default::default(),
            buffer_barriers: Default::default(),
        }
    }
    // adds a special barrier that happens after all the passes in the submission have finished
    // useful for resource queue ownership transfers
    fn reset(&mut self) {
        self.passes.clear();
        self.semaphore_dependencies.clear();
        self.memory_barriers.clear();
        self.image_barriers.clear();
        self.buffer_barriers.clear();
    }
    pub(crate) fn add_final_memory_barrier(&mut self, barrier: MemoryBarrier) {
        self.memory_barriers
            .push((Self::FINAL_SUBMISSION_BARRIER_PASS, barrier));
    }
    pub(crate) fn add_final_image_barrier(&mut self, barrier: ImageBarrier) {
        self.image_barriers
            .push((Self::FINAL_SUBMISSION_BARRIER_PASS, barrier));
    }
    pub(crate) fn add_final_buffer_barrier(&mut self, barrier: BufferBarrier) {
        self.buffer_barriers
            .push((Self::FINAL_SUBMISSION_BARRIER_PASS, barrier));
    }
    pub(crate) fn add_barrier_masks(
        &mut self,
        barrier: CombinedBarrierHandle,
        dst_access: vk::AccessFlags2KHR,
        dst_stages: vk::PipelineStageFlags2KHR,
    ) {
        match barrier.unpack() {
            SubmissionBarrier::Memory(h) => {
                let barrier = &mut self.memory_barriers[h.index()].1;
                barrier.dst_access |= dst_access;
                barrier.dst_stages |= dst_stages;
            }
            SubmissionBarrier::Image(h) => {
                let barrier = &mut self.image_barriers[h.index()].1;
                barrier.dst_access |= dst_access;
                barrier.dst_stages |= dst_stages;
            }
            SubmissionBarrier::Buffer(h) => {
                let barrier = &mut self.buffer_barriers[h.index()].1;
                barrier.dst_access |= dst_access;
                barrier.dst_stages |= dst_stages;
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct OpenSubmision {
    current_pass: Option<TimelinePass>,
    submission: Submission,
}

impl OpenSubmision {
    pub(crate) fn add_pass(&mut self, timeline: TimelinePass, pass: GraphPass) {
        self.current_pass = Some(timeline);
        self.submission.passes.push(pass);
    }
    pub(crate) fn get_current_timeline_pass(&self) -> TimelinePass {
        self.current_pass.unwrap()
    }
    pub(crate) fn get_current_submission_pass(&self) -> SubmissionPass {
        SubmissionPass::new(self.submission.passes.len().checked_sub(1).unwrap())
    }
    pub(crate) fn add_dependency(&mut self, dependency: GraphSubmission) {
        if !self.submission.semaphore_dependencies.contains(&dependency) {
            self.submission.semaphore_dependencies.push(dependency);
        }
    }
    pub(crate) fn finish(&mut self) -> Submission {
        self.submission.clone()
    }
}

pub(crate) struct SubmissionRecorder<'a> {
    graph: &'a GraphCompiler,
    submissions: Vec<Submission>,
    queues: Vec<OpenSubmision>,
    current_queue: Option<(usize, GraphQueue)>,
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
        if let Some(found) = self.find_queue_internal_index(queue) {
            self.current_queue = Some((found.try_into().unwrap(), queue));
        } else {
            let index = self.queues.len();
            self.queues.push(OpenSubmision {
                current_pass: None,
                submission: Submission::new(queue),
            });
            self.current_queue = Some((index.try_into().unwrap(), queue));
        }
    }
    fn find_queue_internal_index(&mut self, queue: GraphQueue) -> Option<usize> {
        self.queues.iter().position(|q| q.submission.queue == queue)
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
                .submission
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
            .map(|found| found.submission.queue)
            .find(|&queue| self.graph.input.get_queue_family(queue) == family)
    }
    /// Adds a closed submission to the structure without disturbing any state
    pub(crate) fn atomic_add_submission(&mut self, submission: Submission) -> GraphSubmission {
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
            self.close_current_submission();
        }
        self.set_current_queue(data.queue);

        let submission = self.get_current_submission();
        let pass_handle = SubmissionPass::new(submission.submission.passes.len());

        let meta = self.graph.get_pass_meta(pass);
        meta.scheduled_submission_position
            .set(OptionalU32::new_some(pass_handle.0));

        let submission = self.get_current_submission_mut();
        submission.add_pass(timeline, pass);

        pass_handle
    }
    pub(crate) fn add_semaphore_dependency(&mut self, dependency: GraphSubmission) {
        self.get_current_submission_mut().add_dependency(dependency);
    }
    // if the pass is from a submission other than the current open one, the submission is closed and the handle to it is returned
    pub(crate) fn prepare_execution_dependency(
        &mut self,
        src_pass: GraphPass,
    ) -> Option<GraphSubmission> {
        let (_, queue) = self.current_queue.unwrap();

        let data = self.graph.input.get_pass_data(src_pass);
        let meta = self.graph.get_pass_meta(src_pass);

        let scheduled = &meta.scheduled_submission.get();

        // the pass is on another queue in an open submission, close it for the next step
        if data.queue != queue && scheduled.is_none() {
            self.close_queue_submission(data.queue)
                .expect("Dependency must be scheduled before the dependee");
        }

        scheduled.get().map(GraphSubmission)
    }
    pub fn close_queue_submission(&mut self, queue: GraphQueue) -> Option<GraphSubmission> {
        let queue_index = self.find_queue_internal_index(queue).unwrap();
        self.__close_submission(queue_index)
    }
    /// False if src_pass is from a separate submission and thus requires no further execution or memory synchronization
    pub(crate) fn add_execution_dependency(&mut self, src_pass: GraphPass) -> bool {
        // the pass has already been submitted, ignore barrier stuff and create a semaphore dependency
        if let Some(scheduled) = self.prepare_execution_dependency(src_pass) {
            let submission = self.get_current_submission_mut();
            submission.add_dependency(scheduled);

            false
        } else {
            true
        }
    }
    pub(crate) fn add_memory_barrier(&mut self, barrier: MemoryBarrier) -> PassMemoryBarrier {
        let c = self.get_current_pass();
        let barriers = &mut self.get_current_submission_mut().submission.memory_barriers;
        let handle = PassMemoryBarrier::new(barriers.len());
        barriers.push((c, barrier));
        handle
    }
    pub(crate) fn add_image_barrier(&mut self, barrier: ImageBarrier) -> PassImageBarrier {
        let c = self.get_current_pass();
        let barriers = &mut self.get_current_submission_mut().submission.image_barriers;
        let handle = PassImageBarrier::new(barriers.len());
        barriers.push((c, barrier));
        handle
    }
    pub(crate) fn add_buffer_barrier(&mut self, barrier: BufferBarrier) -> PassBufferBarrier {
        let c = self.get_current_pass();
        let barriers = &mut self.get_current_submission_mut().submission.buffer_barriers;
        let handle = PassBufferBarrier::new(barriers.len());
        barriers.push((c, barrier));
        handle
    }
    pub(crate) fn close_current_submission(&mut self) {
        self.__close_submission(self.current_queue.unwrap().0);
        self.current_queue = None;
    }
    pub(crate) fn __close_submission(&mut self, queue_index: usize) -> Option<GraphSubmission> {
        let submission = &mut self.queues[queue_index];
        if submission.submission.passes.is_empty() {
            // sanity check
            assert!(submission.submission.memory_barriers.is_empty());
            assert!(submission.submission.image_barriers.is_empty());
            assert!(submission.submission.buffer_barriers.is_empty());
            assert!(submission.submission.semaphore_dependencies.is_empty());

            None
        } else {
            let index: u32 = self.submissions.len().try_into().unwrap();
            let submission_index = OptionalU32::new_some(index);
            for &p in &submission.submission.passes {
                let meta = self.graph.get_pass_meta(p);
                assert!(meta.scheduled_submission.get().is_none());
                meta.scheduled_submission.set(submission_index);
            }

            let finished = submission.finish();
            self.submissions.push(finished);

            submission.submission.reset();

            Some(GraphSubmission(index))
        }
    }
    pub(crate) fn finish(mut self) -> Vec<Submission> {
        // cleanup all open submissions, order doesn't matter since they're leaf nodes
        for queue_i in 0..self.queues.len() {
            self.__close_submission(queue_i);
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
    accessors: SmallVec<[GraphPass; 4]>,
    dst_layout: vk::ImageLayout,
    dst_stages: vk::PipelineStageFlags2KHR,
    dst_access: vk::AccessFlags2KHR,
    dst_queue_family: u32,
}

pub trait ResourceFirstAccessInterface {
    type Accessor: Copy;
    fn accessors(&self) -> &[Self::Accessor];
    fn layout(&self) -> vk::ImageLayout;
    fn stages(&self) -> vk::PipelineStageFlags2KHR;
    fn access(&self) -> vk::AccessFlags2KHR;
    fn queue_family(&self) -> u32;
}

#[derive(Default)]
pub(crate) struct ResourceFirstAccess {
    pub(crate) accessors: SmallVec<[GraphSubmission; 4]>,
    pub(crate) dst_layout: vk::ImageLayout,
    pub(crate) dst_stages: vk::PipelineStageFlags2KHR,
    pub(crate) dst_access: vk::AccessFlags2KHR,
    pub(crate) dst_queue_family: u32,
}

impl ResourceFirstAccessInterface for ResourceFirstAccess {
    type Accessor = GraphSubmission;
    fn accessors(&self) -> &[Self::Accessor] {
        &self.accessors
    }
    fn layout(&self) -> vk::ImageLayout {
        self.dst_layout
    }
    fn stages(&self) -> vk::PipelineStageFlags2KHR {
        self.dst_stages
    }
    fn access(&self) -> vk::AccessFlags2KHR {
        self.dst_access
    }
    fn queue_family(&self) -> u32 {
        self.dst_queue_family
    }
}

pub enum ImageKindCreateInfo<'a> {
    ImageRef(std::cell::Ref<'a, object::ImageCreateInfo>),
    Image(&'a object::ImageCreateInfo),
    Swapchain(&'a object::SwapchainCreateInfo),
}

pub struct GraphCompiler {
    pub(crate) input: CompilationInput,

    pub(crate) pass_objects: Vec<PassObjectState>,

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
            pass_objects: Default::default(),
            pass_meta: Default::default(),
            image_meta: Default::default(),
            buffer_meta: Default::default(),
            alive_passes: Default::default(),
            alive_images: Default::default(),
            alive_buffers: Default::default(),
            pass_children: Default::default(),
            graphics_pipeline_promises: Default::default(),
            function_promises: Default::default(),
            memory: RefCell::new(ManuallyDrop::new(create_memory_suballocators())),
            physical_images: Default::default(),
            physical_buffers: Default::default(),
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
        device.threadpool().spawn(move || {
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
                                let template = unsafe { info.to_vk(&bump) };
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

            sender.send((a, b));
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
                        assert!(
                            dst_writing,
                            "First resource access is a read, this is invalid"
                        );
                        *src = ResourceState::Normal {
                            reading: SmallVec::new(),
                            writing: Some(p),
                        };
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
                        if dst_writing {
                            if let Some(producer) = writing {
                                data.add_dependency(*producer, true, false);
                            }

                            for read in &*reading {
                                data.add_dependency(*read, false, false);
                            }

                            reading.clear();
                            *writing = Some(p);
                        } else {
                            let producer = writing
                                .expect("Resource must first be written to before it is read");
                            data.add_dependency(producer, true, false);

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

        // for (i, p) in self.input.passes.iter().enumerate() {
        //     let pass = GraphPass::new(i);
        //     println!("Pass: {}", self.input.get_pass_display(pass));
        //     print!("  dependencies: ");
        //     let mut first = true;
        //     for d in &p.dependencies {
        //         if !first {
        //             print!(", ");
        //         }
        //         first = false;
        //         let prefix = if d.is_hard() { "!" } else { "" };
        //         print!("{prefix}\"{}\"", self.input.get_pass_display(d.get_pass()))
        //     }
        //     println!();
        // }

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
                    let queue = GraphQueue::new(queue_i);
                    let (position, heap) = &mut available[queue_i];

                    // we've depleted the last flush region, continue to the next one
                    if heap.is_empty() {
                        let mut added_passes = false;
                        let mut added_flush = false;
                        let index = &mut queue_flush_region[queue_i];
                        for &e in &self.input.timeline[*index..] {
                            *index += 1;
                            match e.get() {
                                PassEventData::Pass(next_pass) => {
                                    let data = &self.input.get_pass_data(next_pass);
                                    if data.queue == queue {
                                        let dependency_info =
                                            &mut dependency_count[next_pass.index()];
                                        // if the pass has no outstanding dependencies, we measure its priority and add it to the heap
                                        if dependency_info.0 == 0 {
                                            let _queue = data.queue;
                                            let item = AvailablePass::new(
                                                next_pass,
                                                self,
                                                &mut graph_layers,
                                            );
                                            heap.push(item);
                                        }
                                        dependency_info.1 = true;
                                        added_passes = true;
                                    }
                                }
                                PassEventData::Flush(f) => {
                                    if f == queue {
                                        if added_passes {
                                            // roll back the index, since we wan't this flush to be reprocessed when `scheduled` runs out again
                                            *index = index.checked_sub(1).unwrap();
                                            break;
                                        } else if !added_flush {
                                            scheduled.push(GraphPassEvent::new(
                                                PassEventData::Flush(queue),
                                            ));
                                            added_flush = true;
                                        }
                                    }
                                }
                                // moves are handled later
                                PassEventData::Move(_) => {}
                            }
                        }
                    };

                    let Some(AvailablePass { pass, .. }) = heap.pop() else {
                        continue;
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
        image_moves.sort_by_key(|(p, _)| *p);

        // separate the scheduled passes into specific submissions
        // this is a greedy process, but due to the previous scheduling it should yield somewhat decent results
        let (mut submissions, image_last_state, buffer_last_state, accessors) = {
            let mut image_rw: Vec<ResourceState<ImageMarker>> =
                vec![ResourceState::Uninit; self.input.images.len()];
            let mut buffer_rw: Vec<ResourceState<BufferMarker>> =
                vec![ResourceState::Uninit; self.input.buffers.len()];

            let recorder = RefCell::new(SubmissionRecorder::new(self));

            // set of first accesses that occur in the timeline,
            // becomes "closed" (bool = true) when the first write occurs
            // this will be used during graph execution to synchronize with external resources
            let mut accessors: HashMap<CombinedResourceHandle, ResourceAccessEntry> =
                constant_ahash_hashmap();

            for (timeline_i, &e) in scheduled.iter().enumerate() {
                match e.get() {
                    PassEventData::Pass(p) => {
                        let timeline_pass = e.to_timeline_pass().unwrap();
                        let submission_pass =
                            recorder.borrow_mut().begin_pass(timeline_pass, p, |_| {});

                        let data = self.input.get_pass_data(p);
                        let queue = data.queue;
                        let queue_family = self.input.get_queue_family(queue);

                        for res in &data.images {
                            self.emit_barriers::<ImageMarker>(
                                p,
                                queue_family,
                                res,
                                &recorder,
                                &mut image_rw,
                                &mut accessors,
                            );
                        }

                        for res in &data.buffers {
                            self.emit_barriers::<BufferMarker>(
                                p,
                                queue_family,
                                res,
                                &recorder,
                                &mut buffer_rw,
                                &mut accessors,
                            );
                        }

                        let mut recorder_ref = recorder.borrow_mut();
                        for dep in &data.dependencies {
                            if dep.is_real() {
                                if recorder_ref.add_execution_dependency(dep.get_pass()) {
                                    recorder_ref.add_memory_barrier(MemoryBarrier {
                                        // TODO reconsider if it would be useful to be able to specify more granular dependencies
                                        src_stages: vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                                        dst_stages: data.stages,
                                        src_access: vk::AccessFlags2KHR::MEMORY_WRITE,
                                        dst_access: data.access,
                                    });
                                }
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
                                    ResourceState::Normal(subresource) => {
                                        new_parts.push((src_image, subresource));
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
                    PassEventData::Flush(q) => {
                        recorder.borrow_mut().close_queue_submission(q);
                    }
                }
            }

            let mut submissions = RefCell::into_inner(recorder).finish();

            for sub in &mut submissions {
                // may not be sorted due to later submissions possibly adding barriers willy nilly
                sub.memory_barriers.sort_by_key(|&(p, _)| p);
                sub.image_barriers.sort_by_key(|&(p, _)| p);
                sub.buffer_barriers.sort_by_key(|&(p, _)| p);
            }

            // perform transitive reduction on the submissions
            // lifted from petgraph, https://docs.rs/petgraph/latest/petgraph/algo/tred/fn.dag_transitive_reduction_closure.html
            {
                // make sure that the dependencies are in topological order (ie are sorted)
                for sub in &mut submissions {
                    sub.semaphore_dependencies.sort();
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

            let iter = accessors.into_iter().map(|(key, accessor)| {
                let ResourceAccessEntry { closed, accessors, dst_layout, dst_stages, dst_access, dst_queue_family } = accessor;

                submissions_scratch.clear();
                submissions_scratch.extend(accessors.iter().map(|&pass| {
                    GraphSubmission(self.get_pass_meta(pass).scheduled_submission.get().unwrap())
                }));
                submissions_scratch.sort();
                submissions_scratch.dedup();

                if submissions_scratch.len() == 0 {
                    panic!("Nothing is using the resource? Is this even possible? Wouldn't the resource be dead?");
                }

                let access = ResourceFirstAccess {
                    accessors: SmallVec::from_slice(&submissions_scratch),
                    dst_layout,
                    dst_queue_family,
                    dst_stages,
                    dst_access,
                };

                (key, access)
            });
            final_accessors.extend(iter);

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
            // token, refcount
            let mut after_local_passes: ahash::HashMap<AvailabilityToken, u32> =
                constant_ahash_hashmap();

            for (i, submission) in submissions.iter().enumerate() {
                SubmissionResourceReuse::begin(
                    GraphSubmission::new(i),
                    &submissions,
                    &mut submission_reuse,
                );

                let reuse = &mut submission_reuse[i];

                for (i, &p) in submission.passes.iter().enumerate() {
                    let data = self.input.get_pass_data(p);
                    let d = device.device();

                    need_alloc.clear();
                    for a in &data.images {
                        if let ImageData::TransientPrototype(create_info, _allocation_info) =
                            self.input.get_image_data(a.handle)
                        {
                            let vk_info = create_info.to_vk();
                            let image = unsafe {
                                d.create_image(&vk_info, device.allocator_callbacks())
                                    .unwrap()
                            };
                            maybe_attach_debug_label(
                                image,
                                &self.input.get_image_display(a.handle),
                                &device,
                            );
                            let requirements = unsafe { d.get_image_memory_requirements(image) };
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
                                d.create_buffer(&vk_info, device.allocator_callbacks())
                                    .unwrap()
                            };
                            maybe_attach_debug_label(
                                buffer,
                                &self.input.get_buffer_display(a.handle),
                                &device,
                            );
                            let requirements = unsafe { d.get_buffer_memory_requirements(buffer) };
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
                    need_alloc.sort_by(|(_, a), (_, b)| a.size.cmp(&b.size).reverse());

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
                                reuse,
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
                                    // memory: suballocation,
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
                self.pass_objects[i].create_pass(&mut graph_context);
            }
        }

        for (key, access) in &accessors {
            let &ResourceFirstAccess {
                ref accessors,
                dst_layout,
                dst_stages,
                dst_access,
                dst_queue_family,
            } = access;

            match key.unpack() {
                GraphResource::Image(image) => {
                    let data = self.input.get_image_data(image);
                    let meta = self.get_image_meta(image);

                    match data {
                        ImageData::Swapchain(_) => {
                            assert!(accessors.len() == 1, "Swapchains use legacy semaphores and do not support multiple signals or waits, using a swapchain in multiple submissions is disallowed (you should really just transfer the final image into it at the end of the frame)");

                            let first_access_submission = &mut submissions[accessors[0].index()];

                            if dst_layout != vk::ImageLayout::UNDEFINED {
                                // acquired images start out as UNDEFINED, since we are currently allowing swapchains to only be used in a single submission,
                                // we can just transition the layout at the start of the submission
                                first_access_submission.image_barriers.insert(
                                    0,
                                    (
                                        SubmissionPass(0),
                                        // image,
                                        // src_layout: vk::ImageLayout::UNDEFINED,
                                        // dst_layout,
                                        ImageBarrier {
                                            image,
                                            src_stages: vk::PipelineStageFlags2KHR::empty(),
                                            dst_stages,
                                            src_access: vk::AccessFlags2KHR::empty(),
                                            dst_access,
                                            old_layout: vk::ImageLayout::UNDEFINED,
                                            new_layout: dst_layout,
                                            // we don't care about the contents, no transition is required
                                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                        },
                                    ),
                                )
                            }

                            let src_layout = match &image_last_state[image.index()] {
                                ResourceState::Uninit => {
                                    panic!("Must write to swapchain before presenting")
                                }
                                ResourceState::Normal(ResourceSubresource { layout, .. }) => {
                                    layout.unwrap()
                                }
                                ResourceState::MoveDst { .. } | ResourceState::Moved => {
                                    panic!("Swapchains do not support moving")
                                }
                            };

                            let ResourceState::Normal(ResourceSubresource { read_barrier: src_barrier, layout, queue_family, ref access, last_write } ) = image_last_state[image.index()] else {
                                panic!();
                            };

                            let mut src_stages = vk::PipelineStageFlags2KHR::empty();
                            let mut src_access = vk::AccessFlags2KHR::empty();
                            if access.is_empty() {
                                let touch = last_write.unwrap();
                                src_stages = touch.stages;
                                src_access = touch.access;
                            } else {
                                for touch in access {
                                    src_stages |= touch.stages;
                                }
                            }

                            // swapchain images must be in PRESENT_SRC_KHR before presenting
                            // FIXME if we support moving into external resources (which we should) this will have to be intergrated
                            // with the full barrier emission logic because this is not enough for MoveDst states
                            first_access_submission.add_final_image_barrier(ImageBarrier {
                                image,
                                src_stages,
                                dst_stages: vk::PipelineStageFlags2KHR::empty(),
                                src_access,
                                dst_access: vk::AccessFlags2KHR::empty(),
                                old_layout: src_layout,
                                new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                                // present happens on the same queue as the last access
                                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            });
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        let suballocators = ManuallyDrop::into_inner(RefCell::into_inner(std::mem::replace(
            &mut self.memory,
            RefCell::new(ManuallyDrop::new(create_memory_suballocators())),
        )));

        CompiledGraph {
            state: MainCompiledGraphVulkanState::new(
                device,
                &*suballocators,
                std::mem::take(&mut self.pass_objects),
                std::mem::take(&mut self.physical_images),
                std::mem::take(&mut self.physical_buffers),
                &self.input,
            ),

            input: std::mem::take(&mut self.input),
            timeline: std::mem::take(&mut self.input.timeline),
            submissions,
            external_resource_initial_access: accessors,
            image_last_state,
            buffer_last_state,
            alive_passes: self.alive_passes.take(),
            alive_images: self.alive_images.take(),
            alive_buffers: self.alive_buffers.take(),

            current_generation: Cell::new(None),
            prev_generation: Cell::new(None),
            early_returned: Cell::new(false),
        }
    }

    pub(crate) fn get_last_resource_usage<'a, T: ResourceMarker>(
        &'a self,
        resource_last_state: &'a [ResourceState<T>],
        scratch: &'a mut Vec<(GraphSubmission, SubmissionPass)>,
    ) -> impl Iterator<Item = ResourceLastUse> + 'a
    where
        T::IfImage<vk::ImageLayout>: Clone,
    {
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
                            .flat_map(|(_, subresource)| &subresource.access)
                            .cloned(),
                        scratch,
                        self,
                    );
                }
                ResourceState::Normal(ResourceSubresource { access, .. }) => {
                    add(access.iter().cloned(), scratch, self);
                }
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
                        scratch.sort_by_key(|(sub, _)| *sub);
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
        dst_queue_family: u32,
        resource_data: &T::Data,
        recorder: &RefCell<SubmissionRecorder>,
        resource_rw: &mut [ResourceState<T>],
        accessors: &mut ahash::HashMap<CombinedResourceHandle, ResourceAccessEntry>,
    ) where
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
        let normal_state = |barrier: Option<CombinedBarrierHandle>, dst_writes: bool| {
            if dst_writes {
                ResourceState::Normal(ResourceSubresource {
                    read_barrier: barrier,
                    last_write: Some(dst_touch),
                    layout: dst_layout,
                    queue_family: dst_queue_family,
                    access: SmallVec::new(),
                })
            } else {
                ResourceState::Normal(ResourceSubresource {
                    read_barrier: barrier,
                    last_write: None,
                    layout: dst_layout,
                    queue_family: dst_queue_family,
                    access: smallvec![dst_touch],
                })
            }
        };
        match state {
            // no dependency
            ResourceState::Uninit => {
                if T::IS_IMAGE && !imported && dst_layout.unwrap() != vk::ImageLayout::UNDEFINED {
                    let barrier = recorder.borrow_mut().add_image_barrier(ImageBarrier {
                        image: resource_handle.get_image().unwrap(),
                        src_stages: vk::PipelineStageFlags2KHR::empty(),
                        dst_stages: dst_touch.stages,
                        src_access: vk::AccessFlags2KHR::empty(),
                        dst_access: dst_touch.access,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: dst_layout.unwrap(),
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    });
                    *state = normal_state(Some(barrier.to_combined()), true);
                } else {
                    *state = normal_state(None, dst_writes);
                }
            }
            // if the current access only needs to read, add it to the readers (and handle layout transitions)
            // otherwise synchronize against all passes and transition to normal state
            ResourceState::MoveDst { parts } => {
                assert!(ImageMarker::IS_IMAGE, "Only images can be moved");
                let mut all_parts_written_to = true;

                for (_, subresource) in parts.iter_mut() {
                    let mut dst_writes_copy = dst_writes;

                    self.handle_resource_state_subresource(
                        subresource,
                        dst_touch,
                        dst_queue_family,
                        dst_layout,
                        &mut dst_writes_copy,
                        raw_resource_handle,
                        resource_handle,
                        concurrent,
                        recorder,
                    );

                    if !dst_writes {
                        all_parts_written_to = false;
                    }
                }

                if all_parts_written_to {
                    let mut recorder = recorder.borrow_mut();
                    let submission = &mut recorder.get_current_submission_mut().submission;
                    for (_, subresource) in parts.iter_mut() {
                        if let Some(src_barrier) = subresource.read_barrier {
                            submission.add_barrier_masks(
                                src_barrier,
                                vk::AccessFlags2KHR::empty(),
                                vk::PipelineStageFlags2KHR::TOP_OF_PIPE,
                            );
                        }
                    }
                    // we create a right now useless memory barrier which forms a dependency chain through TOP_OF_PIPE with all the partial source barriers
                    // the masks will be later modified as more passes are processed
                    let barrier = recorder.add_memory_barrier(MemoryBarrier {
                        src_stages: vk::PipelineStageFlags2KHR::TOP_OF_PIPE,
                        dst_stages: vk::PipelineStageFlags2KHR::empty(),
                        src_access: vk::AccessFlags2KHR::empty(),
                        dst_access: vk::AccessFlags2KHR::empty(),
                    });
                    *state = normal_state(Some(barrier.to_combined()), true);
                }
            }
            ResourceState::Normal(subresource) => {
                self.handle_resource_state_subresource(
                    subresource,
                    dst_touch,
                    dst_queue_family,
                    dst_layout,
                    &mut dst_writes,
                    raw_resource_handle,
                    resource_handle,
                    concurrent,
                    recorder,
                );
            }
            // TODO perhaps this shouldn't be a hard error and instead delegate access to the move destination
            ResourceState::Moved => panic!("Attempt to access moved resource"),
        }

        if imported {
            let ResourceAccessEntry {
                closed,
                accessors,
                dst_layout,
                dst_stages,
                dst_access,
                dst_queue_family,
            } = accessors
                .entry(resource_handle)
                .or_insert_with(|| ResourceAccessEntry {
                    closed: false,
                    accessors: smallvec![],
                    dst_layout: dst_layout.to_option().unwrap_or_default(),
                    dst_queue_family,
                    dst_stages: vk::PipelineStageFlags2KHR::empty(),
                    dst_access: vk::AccessFlags2KHR::empty(),
                });

            if !*closed {
                if !dst_writes || accessors.is_empty() {
                    *dst_stages |= dst_touch.stages;
                    *dst_access |= dst_touch.access;
                    if !accessors.contains(&pass) {
                        accessors.push(pass);
                    }
                }
                if dst_writes {
                    *closed = true;
                }
            }
        }
    }
    pub(crate) fn handle_resource_state_subresource<T: ResourceMarker>(
        &self,
        subresource: &mut ResourceSubresource<T>,
        dst_touch: PassTouch,
        dst_queue_family: u32,
        dst_layout: <T as ResourceMarker>::IfImage<vk::ImageLayout>,
        dst_writes: &mut bool,
        raw_resource_handle: RawHandle,
        resource_handle: CombinedResourceHandle,
        concurrent: bool,
        recorder: &RefCell<SubmissionRecorder>,
    ) where
        <T as ResourceMarker>::IfImage<vk::ImageLayout>: Copy,
    {
        let ResourceSubresource {
            read_barrier,
            layout: src_layout,
            queue_family: src_queue_family,
            access,
            last_write,
        } = subresource;

        let mut recorder = recorder.borrow_mut();

        let layout_transition = T::IS_IMAGE
            && dst_layout.unwrap() != vk::ImageLayout::UNDEFINED
            && dst_layout.unwrap() != src_layout.unwrap();

        let queue_ownership_transition = !concurrent
            // if we do not need the contents of the resource (=layout UNDEFINED), we can simply ignore the transfer
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-queue-transfers
            && (T::IS_BUFFER || dst_layout.unwrap() != vk::ImageLayout::UNDEFINED)
            && *src_queue_family != dst_queue_family;

        // do an image or buffer barrier
        if layout_transition || queue_ownership_transition {
            *dst_writes = true;
        }

        if *dst_writes {
            *read_barrier = None;

            let (src_stages, src_access) = if queue_ownership_transition {
                // the last relevant access is are the readers
                if !access.is_empty() {
                    // we need to make a point on the src queue where all of the reads are finished
                    // we will first try to insert a barrier at the end of the submission
                    // otherwise we will create a dummy submission which
                    let mut access_submissions = access
                        .iter()
                        .map(|t| {
                            recorder.prepare_execution_dependency(t.pass).expect(
                                "Queue ownership transition implies non-current source submission",
                            )
                        })
                        .collect::<SmallVec<[_; 16]>>();

                    let single_src_submission = access_submissions[1..]
                        .iter()
                        .all(|&s| s == access_submissions[0]);

                    // if all of the accesses are within the same submission, we can just add a dummy pass at the end
                    // which waits for all of the passes and then releases ownership
                    let (src_submission_handle, src_submission, src_stages, src_access) =
                        if single_src_submission {
                            let src_stages = access
                                .iter()
                                .map(|t| t.stages)
                                .reduce(std::ops::BitOr::bitor)
                                .unwrap();

                            let submission = access_submissions[0];
                            (
                                submission,
                                recorder.get_closed_submission_mut(submission),
                                src_stages,
                                vk::AccessFlags2KHR::empty(),
                            )
                        }
                        // there are multiple submissions which we need to synchronize against to transfer the ownership
                        // the only way to wait for them on a src_queue family is to create a dummy submission which binds them together
                        else {
                            access_submissions.sort();
                            access_submissions.dedup();

                            let queue = recorder.find_queue_with_family(*src_queue_family).unwrap();

                            // TODO merge dummy submissions where possible
                            let submission = Submission {
                                queue,
                                passes: Vec::new(),
                                semaphore_dependencies: access_submissions.into_vec(),
                                memory_barriers: todo!(),
                                image_barriers: todo!(),
                                buffer_barriers: todo!(),
                            };

                            let submission = recorder.atomic_add_submission(submission);
                            (
                                submission,
                                recorder.get_closed_submission_mut(submission),
                                vk::PipelineStageFlags2KHR::empty(),
                                vk::AccessFlags2KHR::empty(),
                            )
                        };

                    // release the ownership
                    match resource_handle.unpack() {
                        GraphResource::Image(image) => {
                            src_submission.add_final_image_barrier(ImageBarrier {
                                image,
                                src_stages,
                                dst_stages: vk::PipelineStageFlags2KHR::empty(),
                                src_access: src_access,
                                dst_access: vk::AccessFlags2KHR::empty(),
                                // the layout will possibly be transitioned in the acquire barrier
                                old_layout: src_layout.unwrap(),
                                new_layout: src_layout.unwrap(),
                                src_queue_family_index: *src_queue_family,
                                dst_queue_family_index: dst_queue_family,
                            });
                        }
                        GraphResource::Buffer(buffer) => {
                            src_submission.add_final_buffer_barrier(BufferBarrier {
                                buffer,
                                src_stages,
                                dst_stages: vk::PipelineStageFlags2KHR::empty(),
                                src_access: src_access,
                                dst_access: vk::AccessFlags2KHR::empty(),
                                src_queue_family_index: *src_queue_family,
                                dst_queue_family_index: dst_queue_family,
                            });
                        }
                    }

                    // depend on the ownership-release-submission
                    recorder.add_semaphore_dependency(src_submission_handle);

                    (
                        vk::PipelineStageFlags2KHR::empty(),
                        vk::AccessFlags2KHR::empty(),
                    )
                } else {
                    let mut stages = vk::PipelineStageFlags2KHR::empty();
                    for touch in access.iter() {
                        if recorder.add_execution_dependency(touch.pass) {
                            stages |= touch.stages;
                        }
                    }

                    (stages, vk::AccessFlags2KHR::empty())
                }
            }
            // last writer (in that case we have a Write-Write)
            else if let Some(touch) = last_write {
                if recorder.add_execution_dependency(touch.pass) {
                    (touch.stages, touch.access)
                } else {
                    (
                        vk::PipelineStageFlags2KHR::empty(),
                        vk::AccessFlags2KHR::empty(),
                    )
                }
            } else {
                panic!("Resource is in Normal state but has no previous writer or readers");
            };

            // check that a barrier is actually neccessary, this may occur if we're the first write for the resource
            if layout_transition
                || queue_ownership_transition
                || !src_access.is_empty()
                || !src_stages.is_empty()
            {
                // either acquire ownership or do a layout transition
                if layout_transition || queue_ownership_transition {
                    // FIXME for ownership transitions: is it more efficient to acquire everything at the start of the submission?
                    // currenly doing it just-in-time seems more debuggable
                    match resource_handle.unpack() {
                        GraphResource::Image(image) => {
                            recorder.add_image_barrier(ImageBarrier {
                                image,
                                // source masks are empty because we're depending on the access through a semaphore
                                src_stages,
                                dst_stages: dst_touch.stages,
                                src_access,
                                dst_access: dst_touch.access,
                                old_layout: src_layout.unwrap(),
                                new_layout: dst_layout.unwrap(),
                                src_queue_family_index: *src_queue_family,
                                dst_queue_family_index: dst_queue_family,
                            });
                        }
                        GraphResource::Buffer(buffer) => {
                            recorder.add_buffer_barrier(BufferBarrier {
                                buffer,
                                src_stages,
                                dst_stages: dst_touch.stages,
                                src_access,
                                dst_access: dst_touch.access,
                                src_queue_family_index: *src_queue_family,
                                dst_queue_family_index: dst_queue_family,
                            });
                        }
                    }
                }
                // otherwise the synchronization examples says that memory barriers are better
                else {
                    recorder.add_memory_barrier(MemoryBarrier {
                        src_stages,
                        dst_stages: dst_touch.stages,
                        src_access,
                        dst_access: dst_touch.access,
                    });
                }
            }

            *last_write = Some(dst_touch);
            access.clear();
        } else {
            let last_write =
                last_write.expect("The resource must be first written to before being read.");

            if recorder.add_execution_dependency(last_write.pass) {
                let read_barrier = read_barrier.get_or_insert_with(|| {
                    let barrier = recorder.add_memory_barrier(MemoryBarrier {
                        src_stages: last_write.stages,
                        dst_stages: vk::PipelineStageFlags2KHR::empty(),
                        src_access: last_write.access,
                        dst_access: vk::AccessFlags2KHR::empty(),
                    });
                    barrier.to_combined()
                });

                recorder
                    .get_current_submission_mut()
                    .submission
                    .add_barrier_masks(*read_barrier, dst_touch.access, dst_touch.stages);
            }

            access.push(dst_touch);
        }

        // update the resource state
        *src_queue_family = dst_queue_family;
        *src_layout = dst_layout;
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
