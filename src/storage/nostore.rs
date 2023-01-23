use std::{mem::ManuallyDrop, ptr::NonNull};

use ahash::HashSet;
use pumice::VulkanResult;

use super::{constant_ahash_randomstate, ArcHeader, MutableShared, ObjectStorage, ReentrantMutex};
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

    unsafe fn get_or_create<'a>(
        &self,
        data: T::InputData<'a>,
        ctx: NonNull<T::Parent>,
    ) -> VulkanResult<ArcHandle<T>> {
        self.lock.with_locked(|lock| {
            T::create(data, ctx.as_ref()).map(|data| {
                let boxed = Box::new(ArcHeader {
                    refcount: 1.into(),
                    object_data: data,
                    storage_data: (),
                    parent: ctx,
                });
                let leak = Box::leak(boxed);
                let handle = ArcHandle(NonNull::from(leak));
                self.handles.get_mut(lock).insert(handle.make_weak_copy());
                handle
            })
        })
    }

    unsafe fn destroy(&self, handle: &ArcHandle<T>) -> VulkanResult<()> {
        let handle = handle.make_weak_copy();
        let alloc = Box::from_raw(handle.get_arc_header_ptr());
        let ArcHeader {
            refcount: _,
            object_data,
            storage_data: _,
            parent,
        } = *alloc;

        self.lock.with_locked(|lock| {
            self.handles.get_mut(lock).remove(&handle);
            T::destroy(&object_data, lock, parent.as_ref())
        })
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
                T::destroy(handle.get_object_data(), lock, handle.get_parent()).unwrap();
            }
        });
    }
}
