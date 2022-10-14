#![allow(unused)]

pub mod inner_context;
pub mod object;
pub mod util;

use std::{
    alloc::Layout,
    cell::RefCell,
    marker::PhantomData,
    mem::ManuallyDrop,
    ptr::{self, NonNull},
    sync::{atomic::AtomicUsize, Arc},
};

use inner_context::InnerContext;
use object::{
    image::ImageInfo,
    storage::{ArcHeader, NoStore, ObjectHeader, ObjectStorage},
};
use pumice::{
    deep_copy::{DeepCopy, DeepCopyBox},
    dumb_hash::DumbHash,
    util::result::VulkanResult,
    vk,
};
use tracing_subscriber::{
    prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt, EnvFilter,
};
use util::format_writer::FormatWriter;

use crate::object::storage::ContextGetStorage;

// macro_rules! arc_handle_impl {
//     ($name:ident: $storage:ident {
//         info = $info_ty:path;
//         handle = $handle_ty:path;
//         create($device:ident, $info:ident) $create_code:tt
//         destroy($device2:ident, $handle:ident) $destroy_code:tt
//     }) => {
//         pub struct $name(ArcHandle<$name, $storage>);
//         impl Object for $name {
//             type CreateInfo = $info_ty;
//             type Handle = $handle_ty;
//             fn create(
//                 $device: &pumice::DeviceWrapper,
//                 $info: &Self::CreateInfo,
//             ) -> VulkanResult<Self::Handle> {
//                 $create_code
//             }
//             fn destroy($device: &pumice::DeviceWrapper, $handle: Self::Handle) -> VulkanResult<()> {
//                 $destroy_code
//             }
//         }
//     };
// }

// arc_handle_impl!(
//     Image: NoStore {
//         info = ImageInfo;
//         handle = vk::Image;
//         create(device, info) {
//             let create_info = vk::ImageCreateInfo {
//                 p_next: ptr::null(),
//                 flags: info.flags,
//                 image_type: info.size.as_image_type(),
//                 format: info.format,
//                 extent: info.size.as_extent_3d(),
//                 mip_levels: info.mip_levels,
//                 array_layers: info.array_layers,
//                 samples: info.samples,
//                 tiling: info.tiling,
//                 usage: info.usage,
//                 // TODO reconsider
//                 sharing_mode: vk::SharingMode::EXCLUSIVE,
//                 queue_family_index_count: 0,
//                 p_queue_family_indices: ptr::null(),
//                 initial_layout: info.initial_layout,
//                 ..Default::default()
//             };
//             // device.create_image(create_info, allocator)
//         }
//         destroy(device, handle) {
//             todo!()
//         }
//     }
// );

// struct Context();

// impl Context {
//     fn create_image(&mut self) -> Image {
//         todo!()
//     }
//     fn create_buffer(&mut self) -> Buffer {
//         todo!()
//     }
//     fn create_graphics_pipeline(&mut self, info: CreateGraphicsPipeline) -> GraphicsPipeline {
//         todo!()
//     }
//     fn create_sampler(&mut self, info: &vk::SamplerCreateInfo) -> Sampler {
//         todo!()
//     }
//     fn create_queue(&mut self) -> Queue {
//         todo!()
//     }
//     fn create_event(&mut self) -> Event {
//         todo!()
//     }
//     fn create_fence(&mut self) -> Fence {
//         todo!()
//     }
//     fn create_semaphore(&mut self) -> Sampler {
//         todo!()
//     }
//     fn create_descriptor_layout(&mut self) -> DescriptorLayout {
//         todo!()
//     }
// }

// pub struct CreateGraphicsPipeline {}

// macro_rules! handle_impl {
//     ($($name:ident),+) => {
//         $(
//             pub struct $name(u32);
//             impl $name {
//                 pub fn new(len: usize) -> Self {
//                     Self(len.try_into().unwrap())
//                 }
//                 pub fn raw(&self) -> usize {
//                     self.0 as usize
//                 }
//             }
//         )+
//     };
// }

// handle_impl!(
//     Image,
//     Buffer,
//     GraphicsPipeline,
//     Sampler,
//     Queue,
//     Event,
//     Fence,
//     DescriptorLayout
// );

fn main() {
    install_tracing_subscriber(None);
}

pub fn install_tracing_subscriber(filter: Option<EnvFilter>) {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(true)
                .with_thread_ids(false)
                .with_target(false)
                .without_time()
                .with_writer(|| FormatWriter::new(std::io::stderr(), "      "))
                .compact(),
        )
        .with(filter.unwrap_or_else(|| EnvFilter::from_default_env()))
        .init();
}
