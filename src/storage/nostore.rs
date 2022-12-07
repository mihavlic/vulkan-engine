use std::{hash::BuildHasher, mem::ManuallyDrop, ptr::NonNull, sync::atomic::Ordering};

use ahash::{HashSet, RandomState};
use pumice::util::result::VulkanResult;

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
                self.handles
                    .borrow_mut(lock)
                    .insert(handle.make_weak_copy());
                handle
            })
        })
    }

    unsafe fn destroy(&self, header: &ArcHandle<T>) {
        let handle = header.make_weak_copy();
        let alloc = Box::from_raw(header.get_arc_header_ptr());
        let ArcHeader { refcount, header } = *alloc;

        self.lock.with_locked(|lock| {
            self.handles.borrow_mut(lock).remove(&handle);
            T::destroy(header.parent(), header.handle, &header.object_data).unwrap();
        });
    }

    fn acquire_exclusive<'a>(&'a self, header: &ArcHandle<T>) -> super::SynchronizationLock<'a> {
        self.lock.lock()
    }

    unsafe fn cleanup(&self) {
        self.lock.with_locked(|lock| {
            for handle in self.handles.borrow_mut(lock).drain() {
                let header = handle.get_header();
                T::destroy(header.parent(), header.handle, &header.object_data).unwrap();
            }
        });
    }
}
