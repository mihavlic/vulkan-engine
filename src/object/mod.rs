mod buffer;
mod image;
mod surface;
mod swapchain;

pub use buffer::*;
pub use image::*;
pub use surface::*;
pub use swapchain::*;

use pumice::VulkanResult;
use std::{borrow::BorrowMut, hash::Hash, mem::ManuallyDrop, ptr::NonNull};

use crate::storage::{ArcHeader, MutableShared, ObjectHeader, ObjectStorage, SynchronizationLock};

pub(crate) trait Object: Sized {
    type CreateInfo;
    type SupplementalInfo;
    type Handle: Copy;
    type Storage: ObjectStorage<Self> + Sync;
    type ObjectData;

    type Parent;

    unsafe fn create(
        ctx: &Self::Parent,
        info: &Self::CreateInfo,
        supplemental_info: &Self::SupplementalInfo,
    ) -> VulkanResult<(Self::Handle, Self::ObjectData)>;

    unsafe fn destroy(
        ctx: &Self::Parent,
        handle: Self::Handle,
        header: &ObjectHeader<Self>,
        lock: &SynchronizationLock,
    ) -> VulkanResult<()>;

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage;
}

pub(crate) struct ArcHandle<T: Object>(pub(crate) NonNull<ArcHeader<T>>);

impl<T: Object> ArcHandle<T> {
    pub unsafe fn get_handle(&self) -> T::Handle {
        self.get_header().handle
    }
    pub unsafe fn get_header(&self) -> &ObjectHeader<T> {
        &self.0.as_ref().header
    }
    pub(crate) unsafe fn get_arc_header(&self) -> &ArcHeader<T> {
        self.0.as_ref()
    }
    pub(crate) unsafe fn get_arc_header_ptr(&self) -> *mut ArcHeader<T> {
        self.0.as_ptr()
    }
    pub(crate) unsafe fn get_storage(&self) -> &T::Storage {
        T::get_storage(self.get_parent())
    }
    pub(crate) unsafe fn get_parent(&self) -> &T::Parent {
        self.get_header().parent()
    }
    pub(crate) unsafe fn make_weak_copy(&self) -> ManuallyDrop<Self> {
        ManuallyDrop::new(std::ptr::read(self))
    }
    pub(crate) unsafe fn access_mutable<
        A,
        B,
        F: FnOnce(&T::ObjectData) -> &MutableShared<A>,
        F2: FnOnce(&mut A) -> B,
    >(
        &self,
        fun1: F,
        fun2: F2,
    ) -> B {
        let storage = self.get_storage();
        let lock = storage.acquire_exclusive(self);

        let mutable = fun1(&self.get_header().object_data);
        let mut refmut = mutable.get_mut(&lock);

        fun2(refmut.borrow_mut())
    }
}

impl<T: Object> PartialEq for ArcHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T: Object> PartialOrd for ArcHandle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl<T: Object> Eq for ArcHandle<T> {}
impl<T: Object> Ord for ArcHandle<T> {
    fn cmp(&self, _other: &Self) -> std::cmp::Ordering {
        todo!()
    }
}
impl<T: Object> Hash for ArcHandle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

pub(crate) struct CloneMany<T: Object> {
    handle: ArcHandle<T>,
    count: usize,
}

impl<T: Object> Iterator for CloneMany<T> {
    type Item = ArcHandle<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.count > 0 {
            self.count -= 1;
            return Some(ArcHandle(self.handle.0));
        }
        return None;
    }
}

impl<T: Object> Drop for CloneMany<T> {
    fn drop(&mut self) {
        // all handles have been reclaimed, no need to update the counter
        if self.count == 0 {
            return;
        }

        unsafe {
            let prev = self
                .handle
                .get_arc_header()
                .refcount
                .fetch_sub(self.count, std::sync::atomic::Ordering::SeqCst);

            assert!(prev > 0);

            // if we just subtracted the same value as was in self.count, self.count is now 0, destroy the object
            if prev == self.count {
                let storage = self.handle.get_storage();
                T::Storage::destroy(storage, &self.handle);
            }
        }
    }
}

impl<T: Object> ArcHandle<T> {
    pub fn clone_many<'a>(&'a self, count: usize) -> CloneMany<T> {
        unsafe {
            let header = self.0;
            let prev = (*header.as_ptr())
                .refcount
                .fetch_add(count, std::sync::atomic::Ordering::SeqCst);

            assert!(prev > 0);

            CloneMany {
                handle: ArcHandle(self.0),
                count,
            }
        }
    }
}

impl<T: Object> Clone for ArcHandle<T> {
    fn clone(&self) -> Self {
        unsafe {
            let header = self.0;
            let prev = (*header.as_ptr())
                .refcount
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            assert!(prev > 0);

            ArcHandle(header)
        }
    }
}

impl<T: Object> Drop for ArcHandle<T> {
    fn drop(&mut self) {
        unsafe {
            let prev = self
                .get_arc_header()
                .refcount
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

            assert!(prev > 0);

            if prev == 1 {
                let storage = self.get_storage();
                T::Storage::destroy(storage, self);
            }
        }
    }
}
