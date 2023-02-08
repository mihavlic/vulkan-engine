use std::{collections::hash_map::Entry, mem::ManuallyDrop, ptr::NonNull};

use ahash::HashSet;
use pumice::VulkanResult;

use super::{
    constant_ahash_hashmap, constant_ahash_hashset, constant_ahash_randomstate,
    unleak_handle_leaked_box, MutableShared, ObjectStorage, ReentrantMutex,
};
use crate::{
    object::{ObjHandle, ObjHeader, ObjRef, Object, ObjectData},
    storage::create_handle,
};

pub(crate) trait ObjectCreateInfoFingerPrint {
    fn get_fingerprint(&self) -> u128;
}

pub(crate) struct InterningStorage<T: Object> {
    lock: ReentrantMutex,
    handles: MutableShared<ahash::HashSet<ManuallyDrop<ObjHandle<T>>>>,
    interned: MutableShared<ahash::HashMap<u128, ManuallyDrop<ObjHandle<T>>>>,
}

impl<T: Object> InterningStorage<T> {
    pub fn new() -> Self {
        InterningStorage {
            lock: ReentrantMutex::new(),
            handles: MutableShared::new(constant_ahash_hashset()),
            interned: MutableShared::new(constant_ahash_hashmap()),
        }
    }
}

impl<T: Object<Storage = Self>> ObjectStorage<T> for InterningStorage<T>
where
    for<'a> T::InputData<'a>: ObjectCreateInfoFingerPrint,
{
    type StorageData = ();

    unsafe fn get_or_create<'a>(
        &self,
        data: T::InputData<'a>,
        ctx: &T::Parent,
    ) -> VulkanResult<ObjHandle<T>> {
        let fingerprint = data.get_fingerprint();
        self.lock.with_locked(|lock| {
            let mut interned_guard = self.interned.get_mut(lock);
            let entry = interned_guard.entry(fingerprint);

            // a handle with the same fingerprint has been already created
            if let Entry::Occupied(occupied) = &entry {
                // we need to check that the handle is actually still alive, otherwise we will need to recreate it
                if self.handles.get_mut(lock).contains(occupied.get()) {
                    return Ok(ManuallyDrop::into_inner(occupied.get().clone()));
                }
            };

            let handle = create_handle(data, (), ctx)?;

            match entry {
                Entry::Occupied(mut o) => {
                    *o.get_mut() = handle.make_weak_copy();
                }
                Entry::Vacant(v) => {
                    v.insert(handle.make_weak_copy());
                }
            }
            let replaced = self.handles.get_mut(lock).insert(handle.make_weak_copy());
            assert!(replaced == false);

            Ok(handle)
        })
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
            self.handles.get_mut(lock).remove(&handle);
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
