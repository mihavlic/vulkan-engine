use std::{mem::ManuallyDrop, ptr::NonNull};

use ahash::HashSet;
use pumice::VulkanResult;

use super::{
    constant_ahash_randomstate, create_header, ArcHeader, MutableShared, ObjectStorage,
    ReentrantMutex,
};
use crate::object::{ArcHandle, Object};

pub(crate) struct SimpleStorage<T: Object> {
    lock: ReentrantMutex,
    handles: MutableShared<HashSet<ManuallyDrop<ArcHandle<T>>>>,
}

impl<T: Object> SimpleStorage<T> {
    pub fn new() -> Self {
        SimpleStorage {
            lock: ReentrantMutex::new(),
            handles: MutableShared::new(HashSet::with_hasher(constant_ahash_randomstate())),
        }
    }
}

impl<T: Object<Storage = Self>> ObjectStorage<T> for SimpleStorage<T> {
    type StorageData = ();

    unsafe fn get_or_create(
        &self,
        info: <T as Object>::CreateInfo,
        supplemental: <T as Object>::SupplementalInfo,
        ctx: NonNull<T::Parent>,
    ) -> VulkanResult<ArcHandle<T>> {
        self.lock.with_locked(|lock| {
            create_header(ctx, info, supplemental, ()).map(|header| {
                let boxed = Box::new(ArcHeader {
                    refcount: 1.into(),
                    header,
                });
                let leak = Box::leak(boxed);
                let handle = ArcHandle(NonNull::from(leak));
                self.handles.get_mut(lock).insert(handle.make_weak_copy());
                handle
            })
        })
    }

    unsafe fn destroy(&self, handle: &ArcHandle<T>) {
        let handle = handle.make_weak_copy();
        let alloc = Box::from_raw(handle.get_arc_header_ptr());
        let ArcHeader {
            refcount: _,
            header,
        } = *alloc;

        self.lock.with_locked(|lock| {
            self.handles.get_mut(lock).remove(&handle);
            T::destroy(header.parent(), header.handle, &header, lock).unwrap();
        });
    }

    fn acquire_exclusive<'a>(&'a self, _handle: &ArcHandle<T>) -> super::SynchronizationLock<'a> {
        self.lock.lock()
    }

    fn acquire_all_exclusive<'a>(&'a self) -> super::SynchronizationLock<'a> {
        self.lock.lock()
    }

    unsafe fn cleanup(&self) {
        self.lock.with_locked(|lock| {
            for handle in self.handles.get_mut(lock).drain() {
                let header = handle.get_header();
                T::destroy(header.parent(), header.handle, &header, lock).unwrap();
            }
        });
    }
}
