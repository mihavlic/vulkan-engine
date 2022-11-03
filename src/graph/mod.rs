use std::{borrow::Cow, cell::Cell, hash::Hash, ops::Deref};

use pumice::{util::result::VulkanResult, vk};
use smallvec::SmallVec;

use crate::{
    arena::uint::{Config, OptionalU32, PackedUint},
    context::device::Device,
    object::{self, Object},
    synchronization,
};

pub trait CreatePass {
    type Pass: Pass;
    fn create(builder: &mut GraphPassBuilder, device: &Device) -> Self::Pass;
}
pub trait Pass {
    fn prepare(&mut self);
    fn execute(self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()>;
}

struct StoredPass<T: Pass>(Option<T>);
trait ObjectSafePass {
    fn prepare(&mut self);
    fn execute(&mut self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()>;
}
impl<T: Pass> ObjectSafePass for StoredPass<T> {
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
    info: object::ImageCreateInfo,
}

pub struct Graph {
    queues: Vec<GraphQueue>,
    passes: Vec<GraphObject<PassData>>,
    images: Vec<GraphObject<ImageData>>,
    buffers: Vec<GraphObject<BufferData>>,

    pass_meta: Vec<PassMeta>,
    image_meta: Vec<ImageMeta>,
    buffer_meta: Vec<BufferMeta>,

    physical_images: Vec<PhysicalImage>,
    physical_buffers: Vec<PhysicalBuffer>,
}

impl Graph {
    fn mark_pass_alive(&self, handle: GraphPass) {
        let i = handle.index();
        let pass = &self.passes[i];
        let meta = &self.pass_meta[i];

        // if it is marked, we have touched its dependencies already and can safely return
        if meta.alive.get() {
            return;
        }

        meta.alive.set(true);
        pass.dependencies().for_each(|p| self.mark_pass_alive(p));
    }
    fn mark_image_alive<'a>(&'a self, mut handle: &'a GraphImage) {
        loop {
            let i = handle.resource() as usize;
            self.image_meta[i].alive.set(true);
            match &*self.images[i] {
                ImageData::Moved { dst, .. } => {
                    handle = dst;
                }
                _ => {}
            }
        }
    }
    fn mark_buffer_alive(&self, handle: &GraphBuffer) {
        let i = handle.resource() as usize;
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
    fn is_image_external<'a>(&'a self, mut handle: &'a GraphImage) -> bool {
        loop {
            match self.get_image_data(handle) {
                ImageData::Transient { .. } => break false,
                ImageData::Imported { .. } => break true,
                ImageData::Swapchain { .. } => break true,
                ImageData::Moved { dst, to } => {
                    handle = dst;
                }
            }
        }
    }
    fn is_buffer_external(&self, mut handle: &GraphBuffer) -> bool {
        match self.get_buffer_data(handle) {
            BufferData::Transient { .. } => false,
            BufferData::Imported { .. } => true,
        }
    }
    fn get_image_data(&self, handle: &GraphImage) -> &ImageData {
        &self.images[handle.resource() as usize]
    }
    fn get_buffer_data(&self, handle: &GraphBuffer) -> &BufferData {
        &self.buffers[handle.resource() as usize]
    }
    fn get_pass_data(&self, handle: GraphPass) -> &PassData {
        &self.passes[handle.0 as usize]
    }
    pub fn run<F: FnOnce(&mut GraphBuilder)>(&mut self, fun: F) {
        self.clear();
        // sound because GraphBuilder is repr(transparent)
        let builder = unsafe { std::mem::transmute::<&mut Graph, &mut GraphBuilder>(self) };
        fun(builder);
        self.prepare_meta();

        for (i, pass) in self.passes.iter().enumerate() {
            if self.pass_meta[i].alive.get() {
                continue;
            }

            if pass
                .images
                .iter()
                .any(|i| self.is_image_external(&i.handle))
                || pass
                    .buffers
                    .iter()
                    .any(|p| self.is_buffer_external(&p.handle))
            {
                self.mark_pass_alive(GraphPass(i as u32));
                for i in &pass.images {
                    self.mark_image_alive(&i.handle);
                }
                for b in &pass.buffers {
                    self.mark_buffer_alive(&b.handle);
                }
            }
        }

        // (some caching where identical frames don't need to be fully recomputed {
        //   while we would like to not schedule the passes every frame, reusing the computation would require checking that would either be extremely fragile
        //   or would require essentially checking for graph isomorphism ... which is almost certainly NP hard
        //   so we will reschedule every frame (possibly add caching for the case the submitted graph is actually completely identical)
        //   and only try to cache vulkan objects, this will lead to some fragmentation so we will have some wasted memory limit or something and then recreate everything
        // })
        //
        // scheduling the graph:
        //
        // walk through the submitted passes and build dependencies between them
        // any pass reading or writing a resource depends on the last writer
        //
        // eliminate dead passes {
        //   get passes that write to global resources or swapchain
        //   mark all of their dependencies
        //   disable all unmarked passes
        // }
        //
        // scheduling {
        //   push passes that have no dependencies
        //   evaluate some cost metric,
        //    - first of all try to minimize the amount of submissions / make passes whose dependees are on other queues get scheduled first
        //    - try to schedule passes with the same dependees together
        //    - schedule the passes whose dependencies have been scheduled most far back first
        //   pick best pass and decrese dependency count of all dependees, push those that reach zero
        //   result is topological sort of passes per queue
        // }
        //
        // asign barriers {
        //   walk through the sorted passes
        //   (make sure that it is possible to just have a single object for resource state)
        //   order between queues should be explicit due to using marked resource handles
        //   when a queue has a pass that is produced by another queue that has not been scheduled, switch to scheduling that queue first
        // }
        //
        // asign physical resources {
        //   create live ranges for resources
        //   figure out memory aliasing
        //   (consider some mechanism which injects additional synchronization to enable more aliasing to respect some memory limit)
        //   (consider implementing another path which caches objects between executions and uses sparse binding to alias resources without having to recreate them every change)
        //   output definitions of all final objects
        //   create the objects (is there any reason to delay their creation? they must be available when recording)
        // }
        //
        // (optionally validate that created barriers are not cyclic as that could maybe happen?)
        //
        // complete barriers for global resources {
        //   lock the synchronization manager and query the state of submissions which last wrote to the global resources
        //   some resources may have already been observed to be safe to read, making some barriers redundant (is memory visibility still guaranteed?), eliminate those (or ignore them at execution time?)
        //   give the manager all the prepared submissions and mark the global resources
        // }
        //
        // (possibly do the global resource barriers lazily on submit to prevent threads submitting work that depends on the possibly long cpu work when recording)
        //
        // record and submit the frame {
        //   all active passes have prepare() called to possibly start executing some async work (TODO consider implementing some better task system)
        //   start going through passes (separate queues may go in paralel?, submitting passes out of execution order is not neccessa) and recording them
        //   (keep track of "current" resource state for passes to query?)
        // }
    }
}

struct PassImageData {
    handle: GraphImage,
    access: vk::AccessFlags2KHR,
    start_layout: vk::ImageLayout,
    end_layout: vk::ImageLayout,
}
struct PassBufferData {
    handle: GraphBuffer,
    access: vk::AccessFlags2KHR,
}

struct PassData {
    queue: GraphQueue,
    images: Vec<PassImageData>,
    buffers: Vec<PassBufferData>,
    dependencies: Vec<GraphPass>,
    pass: Box<dyn ObjectSafePass>,
}
impl PassData {
    fn dependencies<'a>(&'a self) -> impl Iterator<Item = GraphPass> + 'a {
        self.images
            .iter()
            .filter_map(|i| i.handle.producer())
            .chain(self.buffers.iter().filter_map(|b| b.handle.producer()))
            .chain(self.dependencies.iter().cloned())
    }
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
    pub fn import_queue(&mut self, image: synchronization::Queue) -> GraphQueue {
        todo!()
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
    pub fn add_pass<T: CreatePass>(&mut self, queue: GraphQueue, pass: T) -> GraphPass {
        todo!()
    }
    pub fn add_pass_dependency(&mut self, pass: (), depends: ()) {}
}

pub struct GraphPassBuilder;
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
            #[derive(Clone, Copy)]
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

simple_handle! { GraphQueue, GraphPass }

#[derive(Clone, Copy)]
struct ResourceConfig;
impl Config for ResourceConfig {
    // resource index
    const FIRST_BITS: usize = 20;
    // resource producer index, max means no producer
    const SECOND_BITS: usize = 12;
}

#[derive(Clone, Copy)]
struct PassResourceIndex(PackedUint<ResourceConfig, u32>);
impl PassResourceIndex {
    #[inline]
    fn new(resource: u32, producer: Option<u32>) -> Self {
        assert_ne!(producer, Some(ResourceConfig::MAX_SECOND as u32));
        let producer = producer.unwrap_or(ResourceConfig::MAX_SECOND as u32);
        Self(PackedUint::new(resource, producer))
    }
    #[inline]
    fn resource(&self) -> u32 {
        self.0.first()
    }
    #[inline]
    fn producer(&self) -> Option<u32> {
        let value = self.0.second();
        if value == ResourceConfig::MAX_SECOND as u32 {
            None
        } else {
            Some(value)
        }
    }
}

macro_rules! resource_handle {
    ($($name:ident),+) => {
        $(
            pub struct $name(PassResourceIndex);
            impl $name {
                fn new(resource: u32, producer: Option<u32>) -> Self {
                    Self(PassResourceIndex::new(resource, producer))
                }
                fn resource(&self) -> u32 {
                    self.0.resource()
                }
                fn producer(&self) -> Option<GraphPass> {
                    self.0.producer().map(GraphPass)
                }
                /// creates a copy of the handle, bypassing the usual producer-consumer invariants
                fn clone_internal(&self) -> Self {
                    Self(self.0)
                }
            }
        )+
    };
}

resource_handle! { GraphImage, GraphBuffer }
