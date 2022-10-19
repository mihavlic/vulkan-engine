pub mod nostore;

use crate::{
    context::device::InnerDevice,
    object::{ArcHandle, Object},
};
use pumice::util::result::VulkanResult;
use std::{
    cell::UnsafeCell,
    ptr::NonNull,
    sync::{atomic::AtomicUsize, Mutex},
};

pub(crate) trait ObjectStorage<T: Object>: Sized {
    type StorageData;
    unsafe fn get_or_create(
        &self,
        info: <T as Object>::CreateInfo,
        supplemental: <T as Object>::SupplementalInfo,
        ctx: &T::Parent,
    ) -> VulkanResult<ArcHandle<T>>;

    /// when calling this function, exclusive access to the header must be guaranteed
    /// currently this only gets called when the reference count reaches zero which should happen only once
    /// also the reference count cannot decrease while we are holding a mutex?
    unsafe fn destroy(&self, header: *mut ArcHeader<T>);
}

pub(crate) struct ObjectHeader<T: Object> {
    pub(crate) handle: T::Handle,
    pub(crate) info: T::CreateInfo,
    pub(crate) storage_data: <T::Storage as ObjectStorage<T>>::StorageData,
    pub(crate) object_data: <T as Object>::ObjectData,
    parent: NonNull<T::Parent>,
}

impl<T: Object> ObjectHeader<T> {
    pub(crate) unsafe fn parent<'a, 'b>(&'a self) -> &'b T::Parent {
        self.parent.as_ref()
    }
}

#[repr(C)]
pub(crate) struct ArcHeader<T: Object> {
    pub(crate) refcount: AtomicUsize,
    pub(crate) header: ObjectHeader<T>,
}

// help

pub struct MutableShared<T>(UnsafeCell<T>);

impl<T> MutableShared<T> {
    pub(crate) fn new(whalue: T) -> Self {
        MutableShared(UnsafeCell::new(whalue))
    }
    pub unsafe fn borrow_mut(&self) -> &mut T {
        unsafe { &mut *UnsafeCell::get(&self.0) }
    }
}

// aaa

pub struct ReentrantMutex(parking_lot::ReentrantMutex<()>);

impl ReentrantMutex {
    pub fn new() -> Self {
        Self(parking_lot::ReentrantMutex::new(()))
    }
    pub fn with_locked<T, F: FnOnce() -> T>(&self, fun: F) -> T {
        let guard = self.0.lock();
        let out = fun();
        drop(guard);
        out
    }
}

/// # Safety:
/// the storage must be synchronised when calling this method
unsafe fn create_header<T: Object>(
    ctx: &T::Parent,
    info: T::CreateInfo,
    supplemental_info: T::SupplementalInfo,
    storage_data: <T::Storage as ObjectStorage<T>>::StorageData,
) -> VulkanResult<ObjectHeader<T>> {
    T::create(ctx, &info, &supplemental_info).map(|(handle, object_data)| ObjectHeader {
        handle,
        info,
        storage_data,
        object_data,
        parent: NonNull::from(ctx),
    })
}
