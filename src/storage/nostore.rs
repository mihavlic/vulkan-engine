use std::{mem::ManuallyDrop, ptr::NonNull, time::Duration};

use ahash::HashSet;
use pumice::VulkanResult;

use super::{
    constant_ahash_randomstate, create_handle, create_handle_leaked_box, unleak_handle_leaked_box,
    MutableShared, ObjectStorage, ReentrantMutex,
};
use crate::object::{ObjHandle, ObjHeader, ObjRef, Object};

pub struct SimpleStorage<T: Object> {
    lock: ReentrantMutex,
    handles: MutableShared<HashSet<ManuallyDrop<ObjHandle<T>>>>,
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
        ctx: &T::Parent,
    ) -> VulkanResult<ObjHandle<T>> {
        let handle = create_handle(data, (), ctx)?;

        self.lock.with_locked(|lock| {
            let is_true = self.handles.get_mut(lock).insert(handle.make_weak_copy());
            assert!(is_true == true);
        });

        Ok(handle)
    }

    unsafe fn destroy(&self, handle: ManuallyDrop<ObjHandle<T>>) -> VulkanResult<()> {
        let alloc = unleak_handle_leaked_box(handle.make_weak_copy());
        let ObjHeader {
            refcount: _,
            object_data,
            storage_data: _,
            parent,
        } = *alloc;

        self.lock.with_locked(|lock| {
            std::thread::sleep(Duration::from_millis(20));
            let is_true = self.handles.get_mut(lock).remove(&handle);
            assert!(is_true == true, "Double free detected");
            T::destroy(&object_data, lock, parent.as_ref())
        })
    }

    fn acquire_exclusive<'a>(&'a self, _handle: &ObjRef<T>) -> super::SynchronizationLock<'a> {
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

impl<T: Object<Storage = SimpleStorage<T>>> SimpleStorage<T> {
    pub(crate) unsafe fn add_multiple(
        &self,
        datas: Vec<T::Data>,
        ctx: &T::Parent,
    ) -> Vec<ObjHandle<T>> {
        let handles = datas
            .into_iter()
            .map(|data| create_handle_leaked_box(data, (), ctx))
            .collect::<Vec<_>>();

        self.lock.with_locked(|lock| {
            let mut guard = self.handles.get_mut(lock);
            for handle in &handles {
                let is_true = guard.insert(handle.make_weak_copy());
                assert!(is_true == is_true);
            }
        });

        handles
    }
}
