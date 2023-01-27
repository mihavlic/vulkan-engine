mod allocator;
pub mod compile;
pub mod execute;
mod lazy_sorted;
pub mod record;
pub mod resource_marker;
mod reverse_edges;
pub mod task;

use core::panic;
use std::{
    any::Any,
    borrow::{BorrowMut, Cow},
    cell::{Cell, RefCell, RefMut},
    collections::{hash_map::Entry, BinaryHeap},
    fmt::Display,
    hash::Hash,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut, Range},
    sync::{mpsc::channel, Arc},
};

use ahash::HashMap;
use bumpalo::Bump;
use parking_lot::lock_api::RawRwLock;
use pumice::{util::ObjectHandle, vk, vk10::CommandPoolCreateInfo, DeviceWrapper, VulkanResult};
use pumice_vma::AllocationCreateInfo;

use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use slice_group_by::GroupBy;
use smallvec::{smallvec, SmallVec};

use crate::{
    arena::uint::{Config, OptionalU32, PackedUint},
    device::{
        batch::GenerationId,
        inflight::InflightResource,
        submission::{self, QueueSubmission},
        Device, OwnedDevice,
    },
    graph::{
        allocator::MemoryKind,
        resource_marker::{BufferMarker, ImageMarker, TypeOption, TypeSome},
        reverse_edges::{reverse_edges_into, ChildRelativeKey, NodeKey},
        task::GraphicsPipelineResult,
    },
    object::{
        self, raw_info_handle_renderpass, BufferCreateInfo, BufferMutableState,
        ConcreteGraphicsPipeline, GetPipelineResult, GraphicsPipeline, GraphicsPipelineCreateInfo,
        ImageCreateInfo, ImageMutableState, ObjectData, RenderPassMode, SwapchainAcquireStatus,
        SynchronizeResult,
    },
    passes::{CreatePass, RenderPass},
    storage::{constant_ahash_hashmap, constant_ahash_hashset, ObjectStorage},
    token_abuse,
    util::{format_utils::Fun, macro_abuse::WeirdFormatter},
};

use self::{
    allocator::{AvailabilityToken, RcPtrComparator, SuballocationUgh, Suballocator},
    compile::{GraphCompiler, GraphContext},
    record::GraphPassBuilder,
    resource_marker::{ResourceData, ResourceMarker},
    reverse_edges::{DFSCommand, ImmutableGraph, NodeGraph},
    task::{
        CompileGraphicsPipelinesTask, ComputePipelinePromise, ExecuteFnTask, FnPromiseHandle,
        GraphicsPipelineModeEntry, GraphicsPipelinePromise, GraphicsPipelineSrc, Promise, SendAny,
    },
};

struct StoredCreatePass<T: CreatePass>(Cell<Option<(T, T::PreparedData)>>);
impl<T: CreatePass> StoredCreatePass<T> {
    fn new(mut pass: T, builder: &mut GraphPassBuilder) -> Box<dyn ObjectSafeCreatePass> {
        let data = pass.prepare(builder);
        Box::new(Self(Cell::new(Some((pass, data)))))
    }
}
pub(crate) trait ObjectSafeCreatePass: Send {
    fn create(&self, ctx: &mut GraphContext) -> Box<dyn RenderPass + Send>;
}
impl<T: CreatePass> ObjectSafeCreatePass for StoredCreatePass<T> {
    fn create(&self, ctx: &mut GraphContext) -> Box<dyn RenderPass + Send> {
        let (pass, prepared) = Cell::take(&self.0).unwrap();
        pass.create(prepared, ctx)
    }
}

pub struct GraphObject<T> {
    name: Option<Cow<'static, str>>,
    inner: T,
}

impl<T> GraphObject<T> {
    pub(crate) fn get_inner(&self) -> &T {
        &self.inner
    }
    pub(crate) fn get_inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }
    pub(crate) fn display(&self, index: usize) -> GraphObjectDisplay<'_> {
        GraphObjectDisplay {
            name: &self.name,
            index,
            prefix: "#",
        }
    }
    pub(crate) fn map<A, F: FnOnce(T) -> A>(self, fun: F) -> GraphObject<A> {
        GraphObject {
            name: self.name,
            inner: fun(self.inner),
        }
    }
}

impl<T> Deref for GraphObject<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> std::ops::DerefMut for GraphObject<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub struct GraphObjectDisplay<'a> {
    name: &'a Option<Cow<'static, str>>,
    index: usize,
    prefix: &'a str,
}

impl<'a> GraphObjectDisplay<'a> {
    pub(crate) fn set_prefix(mut self, prefix: &'a str) -> Self {
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

pub trait Named<T>: Sized {
    fn to_graph_object(self) -> GraphObject<T>;
}
impl<T> Named<T> for T {
    fn to_graph_object(self) -> GraphObject<T> {
        GraphObject {
            name: None,
            inner: self,
        }
    }
}
impl<T> Named<T> for (T, &'static str) {
    fn to_graph_object(self) -> GraphObject<T> {
        GraphObject {
            name: Some(Cow::Borrowed(self.1)),
            inner: self.0,
        }
    }
}
impl<T> Named<T> for (T, String) {
    fn to_graph_object(self) -> GraphObject<T> {
        GraphObject {
            name: Some(Cow::Owned(self.1)),
            inner: self.0,
        }
    }
}

// #[test]
// fn test_graph() {
//     let device = unsafe { crate::device::__test_init_device(true) };
//     let mut g: GraphCompiler = todo!();
//     g.compile(|b| {
//         let dummy_queue1 = submission::Queue::new(pumice::vk10::Queue::from_raw(1), 0);
//         let dummy_queue2 = submission::Queue::new(pumice::vk10::Queue::from_raw(2), 0);
//         let dummy_queue3 = submission::Queue::new(pumice::vk10::Queue::from_raw(3), 0);

//         let q0 = b.import_queue(dummy_queue1);
//         let q1 = b.import_queue(dummy_queue2);
//         let q2 = b.import_queue(dummy_queue3);

//         let p0 = b.add_pass(q0, |_: &mut GraphPassBuilder, _: &Device| -> () {}, "p0");
//         let p1 = b.add_pass(q0, |_: &mut GraphPassBuilder, _: &Device| {}, "p1");

//         let p2 = b.add_pass(q1, |_: &mut GraphPassBuilder, _: &Device| {}, "p2");
//         let p3 = b.add_pass(q1, |_: &mut GraphPassBuilder, _: &Device| {}, "p3");

//         let p4 = b.add_pass(q2, |_: &mut GraphPassBuilder, _: &Device| {}, "p4");

//         b.add_pass_dependency(p0, p1, true, true);
//         b.add_pass_dependency(p0, p2, true, true);
//         b.add_pass_dependency(p2, p3, true, true);

//         b.add_pass_dependency(p0, p4, true, true);
//         b.add_pass_dependency(p3, p4, true, true);

//         b.force_pass_run(p1);
//         b.force_pass_run(p2);
//         b.force_pass_run(p3);
//         b.force_pass_run(p4);
//     });
// }

#[macro_export]
macro_rules! simple_handle {
    ($($visibility:vis $name:ident),+) => {
        $(
            #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
            #[repr(transparent)]
            $visibility struct $name(u32);
            #[allow(unused)]
            impl $name {
                $visibility fn new(index: usize) -> Self {
                    assert!(index <= u32::MAX as usize);
                    Self(index as u32)
                }
                #[inline]
                $visibility fn index(&self) -> usize {
                    self.0 as usize
                }
                #[inline]
                $visibility fn to_raw(&self) -> $crate::graph::RawHandle {
                    $crate::graph::RawHandle(self.0)
                }
                #[inline]
                $visibility fn from_raw(raw: $crate::graph::RawHandle) -> Self {
                    Self(raw.0)
                }
            }
        )+
    };
}

pub use simple_handle;

simple_handle! {
    pub RawHandle,
    pub GraphQueue, pub GraphPass, pub GraphImage, pub GraphBuffer,
    pub(crate) GraphSubmission, pub(crate) PhysicalImage, pub(crate) PhysicalBuffer, pub(crate) GraphPassMove,
    // like a GraphPassEvent but only ever points to a pass
    pub(crate) TimelinePass,
    // the pass in a submission
    pub(crate) SubmissionPass,
    // vma allocations
    pub(crate) GraphAllocation,
    pub(crate) GraphMemoryTypeHandle
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct MemoryTypeIndex(u32);

macro_rules! optional_index {
    ($($name:ident),+) => {
        $(
            #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
            struct $name(OptionalU32);
            #[allow(unused)]
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
