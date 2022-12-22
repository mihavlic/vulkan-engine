pub mod nostore;

use crate::object::{ArcHandle, Object};
use pumice::VulkanResult;
use std::{
    cell::{RefCell, RefMut, UnsafeCell},
    hash::BuildHasher,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, AtomicPtr, AtomicUsize},
        Mutex,
    },
};

pub enum SynchronizationLock<'a> {
    ReentrantMutexGuard(parking_lot::ReentrantMutexGuard<'a, ()>),
}

pub(crate) struct ObjectRead<'a, T>(NonNull<T>, SynchronizationLock<'a>);

impl<'a, T> ObjectRead<'a, T> {
    pub(crate) fn get_lock(&self) -> &SynchronizationLock<'a> {
        &self.1
    }
}

impl<'a, T> Deref for ObjectRead<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}

pub(crate) trait ObjectStorage<T: Object>: Sized {
    type StorageData;
    unsafe fn get_or_create(
        &self,
        info: <T as Object>::CreateInfo,
        supplemental: <T as Object>::SupplementalInfo,
        ctx: NonNull<T::Parent>,
    ) -> VulkanResult<ArcHandle<T>>;

    unsafe fn destroy(&self, header: &ArcHandle<T>);

    fn acquire_exclusive<'a>(&'a self, header: &ArcHandle<T>) -> SynchronizationLock<'a>;

    fn read_object<'a>(&'a self, header: &ArcHandle<T>) -> ObjectRead<'a, ObjectHeader<T>> {
        let lock = self.acquire_exclusive(header);
        unsafe { ObjectRead(NonNull::from(header.get_header()), lock) }
    }

    unsafe fn cleanup(&self);
}

pub(crate) struct ObjectHeader<T: Object> {
    pub(crate) handle: T::Handle,
    pub(crate) info: T::CreateInfo,
    pub(crate) storage_data: <T::Storage as ObjectStorage<T>>::StorageData,
    pub(crate) object_data: <T as Object>::ObjectData,
    parent: NonNull<T::Parent>,
}

impl<T: Object> ObjectHeader<T> {
    // yep this is so safe
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

pub struct MutableShared<T>(RefCell<T>);

impl<T> MutableShared<T> {
    pub(crate) fn new(value: T) -> Self {
        MutableShared(RefCell::new(value))
    }
    pub unsafe fn borrow_mut<'a>(&'a self, lock: &'a SynchronizationLock) -> RefMut<'a, T> {
        self.0.borrow_mut()
    }
}

unsafe impl<T> Sync for MutableShared<T> {}

// aaa

pub struct ReentrantMutex(parking_lot::ReentrantMutex<()>);

impl ReentrantMutex {
    pub fn new() -> Self {
        Self(parking_lot::ReentrantMutex::new(()))
    }
    pub fn with_locked<T, F: FnOnce(&SynchronizationLock) -> T>(&self, fun: F) -> T {
        let lock = self.lock();
        let out = fun(&lock);
        drop(lock);
        out
    }
    pub fn lock<'a>(&'a self) -> SynchronizationLock<'a> {
        SynchronizationLock::ReentrantMutexGuard(self.0.lock())
    }
}

/// # Safety:
/// the storage must be synchronized when calling this method
unsafe fn create_header<T: Object>(
    ctx: NonNull<T::Parent>,
    info: T::CreateInfo,
    supplemental_info: T::SupplementalInfo,
    storage_data: <T::Storage as ObjectStorage<T>>::StorageData,
) -> VulkanResult<ObjectHeader<T>> {
    T::create(ctx.as_ref(), &info, &supplemental_info).map(|(handle, object_data)| ObjectHeader {
        handle,
        info,
        storage_data,
        object_data,
        parent: ctx,
    })
}

pub(crate) fn constant_ahash_randomstate() -> ahash::RandomState {
    // seed pulled from the crate source
    const PI: [u64; 4] = [
        0x243f_6a88_85a3_08d3,
        0x1319_8a2e_0370_7344,
        0xa409_3822_299f_31d0,
        0x082e_fa98_ec4e_6c89,
    ];
    ahash::RandomState::with_seeds(PI[0], PI[1], PI[2], PI[3])
}

pub(crate) fn constant_ahash_hasher() -> ahash::AHasher {
    constant_ahash_randomstate().build_hasher()
}
