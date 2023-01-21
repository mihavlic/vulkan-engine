pub mod nostore;

use crate::object::{ArcHandle, Object};
use pumice::VulkanResult;
use std::{
    cell::{Ref, RefCell, RefMut},
    hash::BuildHasher,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::atomic::AtomicUsize,
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

pub(crate) struct ObjectReadWrite<'a, T>(NonNull<T>, SynchronizationLock<'a>);

impl<'a, T> ObjectReadWrite<'a, T> {
    pub(crate) fn get_lock(&self) -> &SynchronizationLock<'a> {
        &self.1
    }
}

impl<'a, T> Deref for ObjectReadWrite<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}

impl<'a, T> DerefMut for ObjectReadWrite<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.as_mut() }
    }
}

pub(crate) trait ObjectStorage<T: Object>: Sized {
    type StorageData;
    unsafe fn get_or_create(
        &self,
        data: T::InputData,
        ctx: NonNull<T::Parent>,
    ) -> VulkanResult<ArcHandle<T>>;

    unsafe fn destroy(&self, handle: &ArcHandle<T>) -> VulkanResult<()>;

    // acquires exlusive access for the object pointed to by handle
    fn acquire_exclusive<'a>(&'a self, handle: &ArcHandle<T>) -> SynchronizationLock<'a>;
    // acquires exlusive access for all objects of the
    fn acquire_all_exclusive<'a>(&'a self) -> SynchronizationLock<'a>;

    fn read_object<'a>(&'a self, handle: &ArcHandle<T>) -> ObjectRead<'a, T::Data> {
        let lock = self.acquire_exclusive(handle);
        unsafe { ObjectRead(NonNull::from(handle.get_object_data()), lock) }
    }

    unsafe fn cleanup(&self);
}

// pub(crate) struct ObjectHeader<T: Object> {
//     pub(crate) handle: T::Handle,
//     pub(crate) info: T::CreateInfo,
//     pub(crate) storage_data: <T::Storage as ObjectStorage<T>>::StorageData,
//     pub(crate) object_data: <T as Object>::ObjectData,
//     parent: NonNull<T::Parent>,
// }

// impl<T: Object> ObjectHeader<T> {
//     // yep this is so safe
//     pub(crate) unsafe fn parent<'a, 'b>(&'a self) -> &'b T::Parent {
//         self.parent.as_ref()
//     }
// }

#[repr(C)]
pub(crate) struct ArcHeader<T: Object> {
    pub(crate) refcount: AtomicUsize,
    pub(crate) object_data: T::Data,
    pub(crate) storage_data: <T::Storage as ObjectStorage<T>>::StorageData,
    pub(crate) parent: NonNull<T::Parent>,
}

// help

pub struct MutableShared<T>(RefCell<T>);

impl<T> MutableShared<T> {
    pub(crate) fn new(value: T) -> Self {
        MutableShared(RefCell::new(value))
    }
    pub unsafe fn get<'a>(&'a self, _lock: &'a SynchronizationLock) -> Ref<'a, T> {
        self.0.borrow()
    }
    pub unsafe fn get_mut<'a>(&'a self, _lock: &'a SynchronizationLock) -> RefMut<'a, T> {
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

pub(crate) fn constant_ahash_hashmap<K, V>() -> ahash::HashMap<K, V> {
    ahash::HashMap::with_hasher(constant_ahash_randomstate())
}

pub(crate) fn constant_ahash_hashset<K>() -> ahash::HashSet<K> {
    ahash::HashSet::with_hasher(constant_ahash_randomstate())
}
