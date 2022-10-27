use pumice::vk;

use crate::{
    arena::uint::{Config, PackedUint},
    object::{self, Object},
};

pub trait CreatePass {}
pub trait Pass {}

pub struct Graph;
impl Graph {
    pub fn run<F: FnOnce(&mut GraphBuilder)>(fun: F) {
        let mut builder = GraphBuilder::new();
        fun(&mut builder);
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
    images: Vec<PassImageData>,
    buffers: Vec<PassBufferData>,
}
enum ImageData {
    Swapchain { handle: object::Swapchain },
    Transient { info: object::ImageCreateInfo },
    Moved { dst: GraphImage, to: ImageMoveTo },
}
struct BufferData {
    info: object::BufferCreateInfo,
}

pub struct ImageMoveTo {}

pub struct GraphBuilder {
    passes: Vec<PassData>,
    images: Vec<ImageData>,
    buffers: Vec<BufferData>,
}
impl GraphBuilder {
    fn new() -> Self {
        todo!()
    }
    pub fn acquire_swapchain(&mut self) -> GraphImage {
        todo!()
    }
    pub fn create_image(&mut self) -> GraphImage {
        todo!()
    }
    pub fn create_buffer(&mut self) -> GraphBuffer {
        todo!()
    }
    pub fn move_image(&mut self, src: &GraphImage, dst: &GraphImage, to: ImageMoveTo) {
        todo!()
    }
    pub fn add_pass(&mut self, queue: ()) {}
    pub fn add_pass_dependency(&mut self, pass: (), depends: ()) {}
}

pub struct GraphPassBuilder;
impl GraphPassBuilder {
    pub fn read_image(
        &mut self,
        access: vk::AccessFlags2KHR,
        layout: vk::ImageLayout,
        image: &GraphImage,
    ) {
    }
    pub fn read_buffer(&mut self, access: vk::AccessFlags2KHR, buffer: &GraphBuffer) {}
    pub fn write_image(
        &mut self,
        access: vk::AccessFlags2KHR,
        layout: vk::ImageLayout,
        image: GraphImage,
    ) -> GraphImage {
        todo!()
    }
    pub fn write_buffer(
        &mut self,
        access: vk::AccessFlags2KHR,
        buffer: GraphBuffer,
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

pub struct GraphImage(PassResourceIndex);
pub struct GraphBuffer(PassResourceIndex);

pub struct GraphImageInstance;
pub struct GraphBufferInstance;

#[derive(Clone, Copy)]
struct ResourceConfig;
impl Config for ResourceConfig {
    // resource index
    const FIRST_BITS: usize = 20;
    // resource producer index, zero means no producer
    const SECOND_BITS: usize = 12;
}

struct PassResourceIndex(PackedUint<ResourceConfig, u32>);
impl PassResourceIndex {
    fn new(resource: u32, producer: u32) -> Self {
        Self(PackedUint::new(resource, producer))
    }
    fn resource(&self) -> u32 {
        self.0.first()
    }
    fn producer(&self) -> u32 {
        self.0.second()
    }
}
