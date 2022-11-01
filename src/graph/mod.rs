use std::{borrow::Cow, hash::Hash};

use pumice::{util::result::VulkanResult, vk};

use crate::{
    arena::uint::{Config, PackedUint},
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

pub struct Graph;
impl Graph {
    pub fn run<F: FnOnce(&mut GraphBuilder)>(fun: F) {
        let mut builder = GraphBuilder::new();
        fun(&mut builder);

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
    pass: Box<dyn ObjectSafePass>,
}
enum ImageData {
    Swapchain { handle: object::Swapchain },
    Transient { info: object::ImageCreateInfo },
    Moved { dst: GraphImage, to: ImageMove },
}
struct BufferData {
    info: object::BufferCreateInfo,
}

struct GraphObject<T> {
    name: Option<Cow<'static, str>>,
    inner: T,
}

pub struct ImageMove {
    terget_subresource: vk::ImageSubresourceLayers,
    target_offset: vk::Offset3D,
    extent: vk::Extent3D,
}

pub struct GraphBuilder {
    passes: Vec<GraphObject<PassData>>,
    images: Vec<GraphObject<ImageData>>,
    buffers: Vec<GraphObject<BufferData>>,
}
impl GraphBuilder {
    fn new() -> Self {
        todo!()
    }
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

pub struct GraphExecutor;
impl GraphExecutor {
    pub fn get_image(&self, handle: GraphImage) -> GraphImageInstance {
        todo!()
    }
    pub fn get_buffer(&self, handle: GraphBuffer) -> GraphBufferInstance {
        todo!()
    }
}

#[derive(Clone, Copy)]
pub struct GraphQueue(u32);
#[derive(Clone, Copy)]
pub struct GraphPass(u32);
#[derive(Clone, Copy)]
pub struct GraphImage(PassResourceIndex);
#[derive(Clone, Copy)]
pub struct GraphBuffer(PassResourceIndex);

pub struct GraphImageInstance;
pub struct GraphBufferInstance;

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
    fn new(resource: u32, producer: u32) -> Self {
        Self(PackedUint::new(resource, producer))
    }
    fn resource(&self) -> u32 {
        self.0.first()
    }
    fn producer(&self) -> Option<u32> {
        let value = self.0.second();
        if value == ResourceConfig::MAX_SECOND as u32 {
            None
        } else {
            Some(value)
        }
    }
}
