use std::{
    any::Any,
    cell::UnsafeCell,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use parking_lot::RawRwLock;
use pumice::vk;

use crate::{
    object::{GraphicsPipeline, RenderPassMode},
    simple_handle,
};

pub(crate) struct GraphicsPipelineModeEntry {
    pub(crate) mode: GraphicsPipelineSrc,
    pub(crate) mode_hash: u64,
}

pub(crate) struct CompileGraphicsPipelinesTask {
    pub(crate) pipeline_handle: GraphicsPipeline,
    // this RwLock is currently write locked by mem::forget ing the RwLockWriteGuard
    // TODO probably just use some async executor, this isn't really scalable
    pub(crate) compiling_guard: Option<Arc<RawRwLock>>,
    pub(crate) batch: Vec<GraphicsPipelineModeEntry>,
}

pub(crate) enum GraphicsPipelineSrc {
    Compile(RenderPassMode, u64),
    Wait(Arc<RawRwLock>, u64),
    Ready(vk::Pipeline),
}

pub(crate) enum GraphicsPipelineResult {
    Compile(u64),
    CompiledFinal(vk::Pipeline),
    Wait(Arc<RawRwLock>, u64),
    Ready(vk::Pipeline),
}

pub(crate) struct ExecuteFnTask {
    pub(crate) fun: Box<dyn FnOnce() -> SendAny + Send + Sync>,
}

pub struct Promise<T>(pub(crate) FnPromiseHandle, pub(crate) PhantomData<T>);

#[derive(Clone, Copy)]
pub struct GraphicsPipelinePromise {
    pub(crate) batch_index: u32,
    pub(crate) mode_offset: u32,
}

#[derive(Clone, Copy)]
pub struct ComputePipelinePromise(u32);

simple_handle! {
    pub(crate) FnPromiseHandle
}

pub(crate) struct SendAny(Box<dyn Any>);

impl SendAny {
    pub fn new<T: Send + 'static>(val: T) -> Self {
        Self(Box::new(val))
    }
    pub fn into_any(self) -> Box<dyn Any> {
        self.0
    }
}

unsafe impl Send for SendAny {}

fn is_send_sync<T: Send + Sync>() {}

fn assert_guard_is_send_sync<'a>() {
    // we need to unlock RwLocks on threads other than the one where it was locked because we're criminals
    is_send_sync::<parking_lot::lock_api::RwLockWriteGuard<'a, parking_lot::RawRwLock, ()>>();
}

pub(crate) struct SendSyncUnsafeCell<T: Send + Sync>(UnsafeCell<T>);

impl<T: Send + Sync> SendSyncUnsafeCell<T> {
    pub fn new(val: T) -> Self {
        Self(UnsafeCell::new(val))
    }
    pub unsafe fn get(&self) -> &T {
        &*self.0.get()
    }
    pub unsafe fn get_mut(&self) -> &mut T {
        &mut *self.0.get()
    }
    pub fn into_unsafecell(self) -> UnsafeCell<T> {
        self.0
    }
    pub fn into_inner(self) -> T {
        UnsafeCell::into_inner(self.0)
    }
}

unsafe impl<T: Send + Sync> Send for SendSyncUnsafeCell<T> {}
unsafe impl<T: Send + Sync> Sync for SendSyncUnsafeCell<T> {}

pub(crate) struct SendUnsafeCell<T: Send>(UnsafeCell<T>);

impl<T: Send> SendUnsafeCell<T> {
    pub fn new(val: T) -> Self {
        Self(UnsafeCell::new(val))
    }
    pub unsafe fn get(&self) -> &T {
        &*self.0.get()
    }
    pub unsafe fn get_mut(&self) -> &mut T {
        &mut *self.0.get()
    }
    pub fn into_unsafecell(self) -> UnsafeCell<T> {
        self.0
    }
    pub fn into_inner(self) -> T {
        UnsafeCell::into_inner(self.0)
    }
}

unsafe impl<T: Send> Send for SendUnsafeCell<T> {}
unsafe impl<T: Send> Sync for SendUnsafeCell<T> {}

#[derive(Clone, Copy)]
pub struct UnsafeSend<T>(T);

unsafe impl<T> Send for UnsafeSend<T> {}

impl<T> UnsafeSend<T> {
    pub unsafe fn new(val: T) -> Self {
        Self(val)
    }
    pub fn take(s: Self) -> T {
        s.0
    }
}

impl<T> Deref for UnsafeSend<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for UnsafeSend<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Copy)]
pub struct UnsafeSendSync<T>(T);

unsafe impl<T> Send for UnsafeSendSync<T> {}
unsafe impl<T> Sync for UnsafeSendSync<T> {}

impl<T> UnsafeSendSync<T> {
    pub unsafe fn new(val: T) -> Self {
        Self(val)
    }
    pub fn take(s: Self) -> T {
        s.0
    }
}

impl<T> Deref for UnsafeSendSync<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for UnsafeSendSync<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
