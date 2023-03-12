use std::{borrow::Cow, cell::Cell, marker::PhantomData, sync::Arc};

use parking_lot::lock_api::RawRwLock;
use pumice::vk;
use pumice_vma::AllocationCreateInfo;
use smallvec::SmallVec;

use crate::{
    arena::uint::OptionalU32,
    device::submission,
    object::{self, GetPipelineResult, GraphicsPipeline, ImageCreateInfo, RenderPassMode},
    passes::{CreatePass, RenderPass},
};

use super::{
    compile::{
        BufferMeta, GraphCompiler, GraphContext, GraphPassEvent, ImageMeta, PassDependency,
        PassEventData, PassObjectState,
    },
    task::{
        CompileGraphicsPipelinesTask, ComputePipelinePromise, ExecuteFnTask, FnPromiseHandle,
        GraphicsPipelineModeEntry, GraphicsPipelinePromise, GraphicsPipelineSrc, Promise, SendAny,
    },
    GraphBuffer, GraphImage, GraphObject, GraphObjectDisplay, GraphPass, GraphPassMove, GraphQueue,
    GraphSubmission, Named, ObjectSafeCreatePass, PhysicalBuffer, PhysicalImage, StoredCreatePass,
    SubmissionPass,
};

pub struct PassData {
    pub(crate) queue: GraphQueue,
    pub(crate) force_run: bool,
    pub(crate) images: Vec<PassImageData>,
    pub(crate) buffers: Vec<PassBufferData>,
    pub(crate) stages: vk::PipelineStageFlags2KHR,
    pub(crate) access: vk::AccessFlags2KHR,
    pub(crate) dependencies: Vec<PassDependency>,
}

#[derive(Clone)]
pub struct PassImageData {
    pub(crate) handle: GraphImage,
    pub(crate) stages: vk::PipelineStageFlags2KHR,
    pub(crate) access: vk::AccessFlags2KHR,
    pub(crate) start_layout: vk::ImageLayout,
    pub(crate) end_layout: Option<vk::ImageLayout>,
}

impl PassImageData {
    pub(crate) fn is_written(&self) -> bool {
        self.access.contains_write()
    }
}

pub struct PassBufferData {
    pub(crate) handle: GraphBuffer,
    pub(crate) access: vk::AccessFlags2KHR,
    pub(crate) stages: vk::PipelineStageFlags2KHR,
}

impl PassBufferData {
    pub(crate) fn is_written(&self) -> bool {
        self.access.contains_write()
    }
}

impl PassData {
    pub(crate) fn add_dependency(&mut self, dependency: GraphPass, hard: bool, real: bool) {
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

pub struct ImageMove {
    // we currently only allow moving images of same format and extent
    // and then concatenate their layers in the input order
    pub(crate) from: SmallVec<[GraphImage; 4]>,
    pub(crate) to: GraphImage,
}

#[derive(Clone, Copy)]
pub(crate) struct MovedImageEntry {
    pub(crate) dst: GraphImage,
    pub(crate) dst_layer_offset: u16,
    pub(crate) dst_layer_count: u16,
}

// TODO merge ImageData and BufferData with ResourceMarker
#[derive(Clone)]
pub(crate) enum ImageData {
    // FIXME ImageCreateInfo is large and this scheme is weird
    TransientPrototype(object::ImageCreateInfo, AllocationCreateInfo),
    Transient(PhysicalImage),
    Imported(object::Image),
    Swapchain(object::Swapchain),
    // TODO make GraphImage a Cell and update it when traversing to be a straight reference to the dst image
    // (if the dst image itself has  been moved, we will have to visit its dst image to get the final layer offset and such)
    // dst image, layer offset, layer count
    Moved(Cell<MovedImageEntry>),
}
impl ImageData {
    pub(crate) fn get_variant_name(&self) -> &'static str {
        match self {
            ImageData::TransientPrototype(..) => "TransientPrototype",
            ImageData::Transient(..) => "Transient",
            ImageData::Imported(_) => "Imported",
            ImageData::Swapchain(_) => "Swapchain",
            ImageData::Moved { .. } => "Moved",
        }
    }
    pub(crate) fn is_sharing_concurrent(&self) -> bool {
        match self {
            ImageData::TransientPrototype(info, _) => info.sharing_mode_concurrent,
            ImageData::Imported(handle) => {
                unsafe { handle.0.get_create_info() }.sharing_mode_concurrent
            }
            ImageData::Swapchain(_) => false,
            ImageData::Transient(_) => unreachable!(),
            ImageData::Moved(..) => unreachable!(),
        }
    }
}
pub(crate) enum BufferData {
    TransientPrototype(object::BufferCreateInfo, AllocationCreateInfo),
    Transient(PhysicalBuffer),
    Imported(object::Buffer),
}

impl BufferData {
    pub(crate) fn is_sharing_concurrent(&self) -> bool {
        match self {
            BufferData::TransientPrototype(info, _) => info.sharing_mode_concurrent,
            BufferData::Transient(_) => unreachable!(),
            BufferData::Imported(handle) => {
                unsafe { handle.0.get_create_info() }.sharing_mode_concurrent
            }
        }
    }
}

#[derive(Default)]
pub struct CompilationInput {
    pub(crate) queues: Vec<GraphObject<submission::Queue>>,
    pub(crate) images: Vec<GraphObject<ImageData>>,
    pub(crate) buffers: Vec<GraphObject<BufferData>>,

    pub(crate) passes: Vec<GraphObject<PassData>>,
    pub(crate) moves: Vec<ImageMove>,

    pub(crate) timeline: Vec<GraphPassEvent>,
}

impl CompilationInput {
    pub(crate) fn clear(&mut self) {
        self.queues.clear();
        self.images.clear();
        self.buffers.clear();
        self.timeline.clear();
        self.passes.clear();
        self.moves.clear();
    }
    pub(crate) fn get_image_data(&self, image: GraphImage) -> &ImageData {
        &self.images[image.index()]
    }
    pub(crate) fn get_image_data_mut(&mut self, image: GraphImage) -> &mut ImageData {
        &mut self.images[image.index()]
    }
    pub(crate) fn get_buffer_data(&self, buffer: GraphBuffer) -> &BufferData {
        &self.buffers[buffer.index()]
    }
    pub(crate) fn get_buffer_data_mut(&mut self, buffer: GraphBuffer) -> &mut BufferData {
        &mut self.buffers[buffer.index()]
    }
    pub(crate) fn get_pass_data(&self, pass: GraphPass) -> &PassData {
        &self.passes[pass.0 as usize]
    }
    pub(crate) fn get_pass_move(&self, move_handle: GraphPassMove) -> &ImageMove {
        &self.moves[move_handle.index()]
    }
    pub(crate) fn get_dependencies(&self, pass: GraphPass) -> &[PassDependency] {
        &self.passes[pass.index()].dependencies
    }
    pub(crate) fn get_queue_family(&self, queue: GraphQueue) -> u32 {
        self.queues[queue.index()].family()
    }
    pub(crate) fn get_queue_families(&self) -> SmallVec<[u32; 4]> {
        let mut vec: SmallVec<[u32; 4]> = SmallVec::new();
        for queue in &self.queues {
            vec.push(queue.family());
        }
        vec.sort_unstable();
        vec.dedup();
        vec
    }
    pub(crate) fn get_queue_display(&self, queue: GraphQueue) -> GraphObjectDisplay<'_> {
        self.queues[queue.index()]
            .display(queue.index())
            .set_prefix("queue")
    }
    pub(crate) fn get_pass_display(&self, pass: GraphPass) -> GraphObjectDisplay<'_> {
        self.passes[pass.index()]
            .display(pass.index())
            .set_prefix("pass")
    }
    pub(crate) fn get_image_display(&self, image: GraphImage) -> GraphObjectDisplay<'_> {
        self.images[image.index()]
            .display(image.index())
            .set_prefix("image")
    }
    pub(crate) fn get_buffer_display(&self, buffer: GraphBuffer) -> GraphObjectDisplay<'_> {
        self.buffers[buffer.index()]
            .display(buffer.index())
            .set_prefix("buffer")
    }
    pub(crate) fn get_concrete_image_data(&self, image: GraphImage) -> &ImageData {
        self.get_concrete_image_data_impl(image).1
    }
    pub(crate) fn get_concrete_image_handle(&self, image: GraphImage) -> GraphImage {
        self.get_concrete_image_data_impl(image).0.dst
    }
    fn get_concrete_image_data_impl(&self, image: GraphImage) -> (MovedImageEntry, &ImageData) {
        let mut image = image;
        loop {
            match self.get_image_data(image) {
                ImageData::Moved(old_entry) => {
                    let mut entry = old_entry.get();

                    let (target_entry, data) = self.get_concrete_image_data_impl(entry.dst);

                    entry.dst_layer_offset += target_entry.dst_layer_offset;
                    entry.dst = target_entry.dst;
                    old_entry.set(entry);

                    return (entry, data);
                }
                other => {
                    return (
                        MovedImageEntry {
                            dst: image,
                            dst_layer_offset: 0,
                            dst_layer_count: 0,
                        },
                        other,
                    );
                }
            }
        }
    }
    pub(crate) fn get_image_subresource_layer_offset(&self, image: GraphImage) -> u32 {
        // we're calling this for the side effect of short circuiting all image move sequences (there's interior mutability)
        let _ = self.get_concrete_image_data(image);
        if let ImageData::Moved(entry) = self.get_image_data(image) {
            entry.get().dst_layer_offset as u32
        } else {
            0
        }
    }
    /// Returns the base array layer in the image and the count of following layers, None if it's the whole resource
    pub(crate) fn get_image_subresource_layer_offset_count(
        &self,
        image: GraphImage,
    ) -> (u32, Option<u32>) {
        // we're calling this for the side effect of short circuiting all image move sequences (there's interior mutability)
        let _ = self.get_concrete_image_data(image);
        if let ImageData::Moved(entry) = self.get_image_data(image) {
            let entry = entry.get();
            (
                entry.dst_layer_offset as u32,
                Some(entry.dst_layer_count as u32),
            )
        } else {
            (0, None)
        }
    }
    pub fn get_image_subresource_range(
        &self,
        image: GraphImage,
        aspect: vk::ImageAspectFlags,
    ) -> vk::ImageSubresourceRange {
        let (base_array_layer, layer_count) = self.get_image_subresource_layer_offset_count(image);
        vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer,
            layer_count: layer_count.unwrap_or(vk::REMAINING_ARRAY_LAYERS),
        }
    }
    pub fn get_image_subresource_layers(
        &self,
        image: GraphImage,
        aspect: vk::ImageAspectFlags,
        mip_level: u32,
    ) -> vk::ImageSubresourceLayers {
        let (base_array_layer, layer_count) = self.get_image_subresource_layer_offset_count(image);
        vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level,
            base_array_layer,
            layer_count: layer_count.unwrap_or(vk::REMAINING_ARRAY_LAYERS),
        }
    }
}

#[repr(transparent)]
pub struct GraphBuilder(GraphCompiler);

impl GraphBuilder {
    pub fn acquire_swapchain(&mut self, swapchain: impl Named<object::Swapchain>) -> GraphImage {
        let handle = GraphImage::new(self.0.input.images.len());
        self.0
            .input
            .images
            .push(Named::to_graph_object(swapchain).map(ImageData::Swapchain));
        handle
    }
    pub fn import_queue(&mut self, queue: impl Named<submission::Queue>) -> GraphQueue {
        let queue = Named::to_graph_object(queue);
        if let Some(i) = self
            .0
            .input
            .queues
            .iter()
            .position(|q| q.inner.raw() == queue.inner.raw())
        {
            GraphQueue::new(i)
        } else {
            let handle = GraphQueue::new(self.0.input.queues.len());
            self.0.input.queues.push(queue);
            handle
        }
    }
    pub fn import_image(&mut self, image: impl Named<object::Image>) -> GraphImage {
        let handle = GraphImage::new(self.0.input.images.len());
        self.0
            .input
            .images
            .push(Named::to_graph_object(image).map(ImageData::Imported));
        handle
    }
    pub fn import_buffer(&mut self, buffer: impl Named<object::Buffer>) -> GraphBuffer {
        let handle = GraphBuffer::new(self.0.input.buffers.len());
        self.0
            .input
            .buffers
            .push(Named::to_graph_object(buffer).map(BufferData::Imported));
        handle
    }
    pub fn create_image(
        &mut self,
        info: object::ImageCreateInfo,
        allocation: pumice_vma::AllocationCreateInfo,
    ) -> GraphImage {
        let handle = GraphImage::new(self.0.input.images.len());
        self.0.input.images.push(GraphObject {
            name: info.label.clone(),
            inner: ImageData::TransientPrototype(info, allocation),
        });
        handle
    }
    pub fn create_buffer(
        &mut self,
        info: object::BufferCreateInfo,
        allocation: pumice_vma::AllocationCreateInfo,
    ) -> GraphBuffer {
        let handle = GraphBuffer::new(self.0.input.buffers.len());
        self.0.input.buffers.push(GraphObject {
            name: info.label.clone(),
            inner: BufferData::TransientPrototype(info, allocation),
        });
        handle
    }
    #[track_caller]
    pub fn move_image<T: IntoIterator<Item = GraphImage>>(
        &mut self,
        images: impl Named<T>,
    ) -> GraphImage {
        let image = GraphImage::new(self.0.input.images.len());

        let object = Named::to_graph_object(images).map(|a| {
            let images = a.into_iter().collect::<SmallVec<_>>();

            // TODO perhaps it would be useful to move into already instantiated non-transient images
            let invalid_data_panic = |image: GraphImage, data: &ImageData| {
                panic!(
                    "Only Transient images can be moved, image '{}' has state '{}'",
                    self.0.input.get_image_display(image),
                    data.get_variant_name()
                )
            };

            let ImageData::TransientPrototype(mut first_info, mut first_allocation) = self.0.input.get_image_data(images[0]).clone() else {
                invalid_data_panic(images[0], self.0.input.get_image_data(images[0]))
            };

            // check that all of them are transient and that they have the same format and extent
            for &i in &images[1..] {
                let data = &self.0.input.get_image_data(i);
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
                let ImageData::TransientPrototype(info, _) = self.0.input.get_image_data(i) else {
                    unreachable!()
                };

                let layer_count: u16 = info.array_layers.try_into().unwrap();
                *self.0.input.images[i.index()].get_inner_mut() =
                    ImageData::Moved(Cell::new(MovedImageEntry {
                        dst: image,
                        dst_layer_offset: layer_offset,
                        dst_layer_count: layer_count,
                    }));
                layer_offset += layer_count;
            }

            let info = ImageCreateInfo {
                array_layers: layer_offset as u32,
                ..first_info
            };

            self.0.input
                .timeline
                .push(GraphPassEvent::new(PassEventData::Move(
                    super::GraphPassMove::new(self.0.input.moves.len()),
                )));
            self.0.input.moves.push(ImageMove {
                from: images,
                to: image,
            });

            ImageData::TransientPrototype(info, first_allocation)
        });

        self.0.input.images.push(object);

        image
    }
    pub fn add_pass<T: CreatePass, N: Into<Cow<'static, str>>>(
        &mut self,
        queue: GraphQueue,
        pass: T,
        name: N,
    ) -> GraphPass {
        let handle = GraphPass::new(self.0.input.passes.len());
        let (data, pass_object) = {
            let mut builder = GraphPassBuilder::new(self, handle);
            let pass_object = StoredCreatePass::new(pass, &mut builder);
            (builder.finish(queue), pass_object)
        };
        self.0
            .input
            .passes
            .push(GraphObject::from_cow(name.into(), data));
        self.0
            .pass_objects
            .push(PassObjectState::Initial(pass_object));
        self.0
            .input
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
        self.0.input.passes[then.index()].add_dependency(first, hard, real);
    }
    pub fn force_pass_run(&mut self, pass: GraphPass) {
        self.0.input.passes[pass.index()].force_run = true;
    }
    pub fn add_scheduling_barrier(&mut self, queue: GraphQueue) {
        self.0
            .input
            .timeline
            .push(GraphPassEvent::new(PassEventData::Flush(queue)));
    }
}

pub struct GraphPassBuilder<'a> {
    graph_builder: &'a mut GraphBuilder,
    pass: GraphPass,
    images: Vec<PassImageData>,
    buffers: Vec<PassBufferData>,
    dependencies: Vec<PassDependency>,
}

impl<'a> GraphPassBuilder<'a> {
    pub(crate) fn new(graph_builder: &'a mut GraphBuilder, pass: GraphPass) -> Self {
        Self {
            graph_builder,
            pass,
            images: Vec::new(),
            buffers: Vec::new(),
            dependencies: Vec::new(),
        }
    }
    pub fn compile_graphics_pipeline(
        &mut self,
        pipeline: &GraphicsPipeline,
        mode: &RenderPassMode,
    ) -> GraphicsPipelinePromise {
        let mode_hash = mode.get_hash();
        let mut promises = &mut self.graph_builder.0.graphics_pipeline_promises;

        // TODO do better than linear search
        if let Some(found) = promises.iter().position(|b| &b.pipeline_handle == pipeline) {
            let CompileGraphicsPipelinesTask {
                pipeline_handle,
                compiling_guard,
                batch,
            } = &mut promises[found];

            if let Some(found_offset) = batch.iter().position(|b| b.mode_hash == mode_hash) {
                GraphicsPipelinePromise {
                    batch_index: found as u32,
                    mode_offset: found_offset as u32,
                }
            } else {
                let handle = GraphicsPipelinePromise {
                    batch_index: found as u32,
                    mode_offset: batch.len() as u32,
                };

                let lock = || {
                    if let Some(lock) = compiling_guard {
                        lock.clone()
                    } else {
                        let lock = Arc::new(parking_lot::RawRwLock::INIT);
                        assert!(lock.try_lock_exclusive());

                        *compiling_guard = Some(lock.clone());

                        lock
                    }
                };

                let result = unsafe {
                    pipeline
                        .0
                        .access_mutable(|d| &d.mutable, |m| m.get_pipeline(mode_hash, lock))
                };

                let src = match result {
                    GetPipelineResult::Ready(ok) => GraphicsPipelineSrc::Ready(ok),
                    GetPipelineResult::Promised(lock) => GraphicsPipelineSrc::Wait(lock, mode_hash),
                    GetPipelineResult::MustCreate(lock) => {
                        GraphicsPipelineSrc::Compile(mode.clone(), mode_hash)
                    }
                };

                batch.push(GraphicsPipelineModeEntry {
                    mode: src,
                    mode_hash,
                });

                handle
            }
        } else {
            let handle = GraphicsPipelinePromise {
                batch_index: promises.len() as u32,
                mode_offset: 0,
            };

            let lock = || {
                let lock = Arc::new(parking_lot::RawRwLock::INIT);
                assert!(lock.try_lock_exclusive());
                lock
            };

            let result = unsafe {
                pipeline
                    .0
                    .access_mutable(|d| &d.mutable, |m| m.get_pipeline(mode_hash, lock))
            };

            let (src, lock) = match result {
                GetPipelineResult::Ready(ok) => (GraphicsPipelineSrc::Ready(ok), None),
                GetPipelineResult::Promised(lock) => {
                    (GraphicsPipelineSrc::Wait(lock, mode_hash), None)
                }
                GetPipelineResult::MustCreate(lock) => (
                    GraphicsPipelineSrc::Compile(mode.clone(), mode_hash),
                    Some(lock),
                ),
            };

            promises.push(CompileGraphicsPipelinesTask {
                pipeline_handle: pipeline.clone(),
                compiling_guard: lock,
                batch: vec![GraphicsPipelineModeEntry {
                    mode: src,
                    mode_hash,
                }],
            });
            handle
        }
    }
    pub fn compile_compute_pipeline(&mut self) -> ComputePipelinePromise {
        todo!()
    }
    pub fn run_task<T: 'static + Send, F: FnOnce() -> T + Send + Sync + 'static>(
        &mut self,
        fun: F,
    ) -> Promise<T> {
        let mut promises = &mut self.graph_builder.0.function_promises;
        let handle = FnPromiseHandle::new(promises.len());

        promises.push(ExecuteFnTask {
            fun: Box::new(move || SendAny::new(fun())),
        });

        Promise(handle, PhantomData)
    }
    pub fn get_image_format(&self, image: GraphImage) -> vk::Format {
        let data = self.graph_builder.0.input.get_image_data(image);
        match data {
            ImageData::TransientPrototype(info, _) => info.format,
            ImageData::Transient(_) => panic!(),
            ImageData::Imported(handle) => unsafe { handle.0.get_create_info().format },
            ImageData::Swapchain(handle) => unsafe { handle.0.get_create_info().format },
            ImageData::Moved(_) => panic!(),
        }
    }
    // TODO check (possibly update for transients) usage flags against their create info
    #[track_caller]
    pub fn use_image(
        &mut self,
        image: GraphImage,
        usage: vk::ImageUsageFlags,
        stages: vk::PipelineStageFlags2KHR,
        access: vk::AccessFlags2KHR,
        start_layout: vk::ImageLayout,
        end_layout: Option<vk::ImageLayout>,
    ) {
        let data = self.graph_builder.0.input.get_image_data(image);
        let resource_usage = match data {
            ImageData::TransientPrototype(info, _) => info.usage,
            ImageData::Imported(archandle) => unsafe { archandle.0.get_create_info().usage },
            ImageData::Swapchain(archandle) => unsafe { archandle.0.get_create_info().usage },
            ImageData::Transient(_) | ImageData::Moved(_) => unreachable!(),
        };

        assert!(
            resource_usage.contains(usage),
            "Resource '{}' is missing usage {:?}",
            self.graph_builder.0.input.get_image_display(image),
            usage
        );

        // TODO deduplicate or explicitly forbid multiple entries with the same handle
        self.images.push(PassImageData {
            handle: image,
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
        let data = self.graph_builder.0.input.get_buffer_data(buffer);
        let resource_usage = match data {
            BufferData::TransientPrototype(info, _) => info.usage,
            BufferData::Imported(archandle) => unsafe { archandle.0.get_create_info().usage },
            BufferData::Transient(_) => unreachable!(),
        };

        assert!(
            resource_usage.contains(usage),
            "Resource '{}' is missing usage {:?}",
            self.graph_builder.0.input.get_buffer_display(buffer),
            usage
        );

        // TODO deduplicate or explicitly forbid multiple entries with the same handle
        self.buffers.push(PassBufferData {
            handle: buffer,
            access,
            stages,
        });
    }
    fn finish(self, queue: GraphQueue) -> PassData {
        let mut access = vk::AccessFlags2KHR::default();
        let mut stages = vk::PipelineStageFlags2KHR::default();
        for i in &self.images {
            access |= i.access;
            stages |= i.stages;
        }
        for b in &self.buffers {
            access |= b.access;
            stages |= b.stages;
        }

        PassData {
            queue: queue,
            force_run: false,
            images: self.images,
            buffers: self.buffers,
            stages,
            access,
            dependencies: self.dependencies,
        }
    }
}
