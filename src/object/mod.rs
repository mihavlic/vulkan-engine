macro_rules! create_object {
    ($name:ident) => {
        #[derive(Clone, PartialEq, Eq)]
        pub struct $name(pub(crate) $crate::object::ArcHandle<Self>);

        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}
    };
}

macro_rules! derive_raw_handle {
    ($name:ident, $Handle:ty) => {
        impl $name {
            pub fn raw(&self) -> $Handle {
                unsafe { self.0.get_handle() }
            }
        }
    };
}

mod buffer;
mod descriptor_set_layout;
mod graphics_pipeline;
mod image;
mod pipeline_layout;
mod render_pass;
mod sampler;
mod shader_module;
mod surface;
mod swapchain;

pub use buffer::*;
pub use descriptor_set_layout::*;
pub use graphics_pipeline::*;
pub use image::*;
pub use pipeline_layout::*;
pub use render_pass::*;
pub use sampler::*;
pub use shader_module::*;
pub use surface::*;
pub use swapchain::*;

use pumice::VulkanResult;
use std::{borrow::BorrowMut, hash::Hash, mem::ManuallyDrop, ptr::NonNull};

use crate::storage::{ArcHeader, MutableShared, ObjectStorage, SynchronizationLock};

pub(crate) trait ObjectData {
    type CreateInfo;
    type Handle: Copy;
    /// The method must be safe to be called from multiple threads at the same time
    fn get_create_info(&self) -> &Self::CreateInfo;
    /// The method must be safe to be called from multiple threads at the same time
    fn get_handle(&self) -> Self::Handle;
}

pub(crate) trait Object: Sized {
    type Storage: ObjectStorage<Self> + Sync;
    type Parent;

    type InputData<'b>;
    type Data: ObjectData;

    unsafe fn create<'a>(data: Self::InputData<'a>, ctx: &Self::Parent)
        -> VulkanResult<Self::Data>;

    unsafe fn destroy(
        data: &Self::Data,
        lock: &SynchronizationLock,
        ctx: &Self::Parent,
    ) -> VulkanResult<()>;

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage;
}

pub(crate) struct ArcHandle<T: Object>(pub(crate) NonNull<ArcHeader<T>>);

/// Pointed-to object is atomically reference counted, fields are either immutable or synchronized with MutableShared
unsafe impl<T: Object + Send> Send for ArcHandle<T> {}
unsafe impl<T: Object + Sync> Sync for ArcHandle<T> {}

impl<T: Object> ArcHandle<T> {
    pub(crate) unsafe fn get_object_data(&self) -> &T::Data {
        &self.0.as_ref().object_data
    }
    pub(crate) unsafe fn get_storage_data(&self) -> &<T::Storage as ObjectStorage<T>>::StorageData {
        &self.0.as_ref().storage_data
    }
    pub(crate) unsafe fn get_parent_storage(&self) -> &T::Storage {
        let parent = self.get_parent();
        T::get_storage(parent)
    }
    pub(crate) unsafe fn get_handle(&self) -> <T::Data as ObjectData>::Handle {
        self.get_object_data().get_handle()
    }
    pub(crate) unsafe fn get_create_info(&self) -> &<T::Data as ObjectData>::CreateInfo {
        self.get_object_data().get_create_info()
    }
    pub(crate) unsafe fn get_arc_header(&self) -> &ArcHeader<T> {
        self.0.as_ref()
    }
    pub(crate) unsafe fn get_arc_header_ptr(&self) -> *mut ArcHeader<T> {
        self.0.as_ptr()
    }
    pub(crate) unsafe fn get_parent(&self) -> &T::Parent {
        self.get_arc_header().parent.as_ref()
    }
    pub(crate) unsafe fn make_weak_copy(&self) -> ManuallyDrop<Self> {
        ManuallyDrop::new(std::ptr::read(self))
    }
    pub(crate) unsafe fn lock_storage(&self) -> SynchronizationLock<'_> {
        let storage = self.get_parent_storage();
        storage.acquire_exclusive(self)
    }
    pub(crate) unsafe fn access_mutable<
        A,
        B,
        F: FnOnce(&T::Data) -> &MutableShared<A>,
        F2: FnOnce(&mut A) -> B,
    >(
        &self,
        fun1: F,
        fun2: F2,
    ) -> B {
        let lock = self.lock_storage();

        let mutable = fun1(&self.get_object_data());
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
                let storage = self.handle.get_parent_storage();
                T::Storage::destroy(storage, &self.handle).unwrap();
            }
        }
    }
}

impl<T: Object> ArcHandle<T> {
    pub(crate) fn clone_many<'a>(&'a self, count: usize) -> CloneMany<T> {
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
                let storage = self.get_parent_storage();
                T::Storage::destroy(storage, self).unwrap();
            }
        }
    }
}

pub(crate) struct BasicObjectData<H: Copy, I> {
    handle: H,
    info: I,
}

impl<H: Copy, I> BasicObjectData<H, I> {
    fn new_result(handle: VulkanResult<H>, info: I) -> VulkanResult<Self> {
        handle.map(|handle| Self { handle, info })
    }
    fn new(handle: H, info: I) -> VulkanResult<Self> {
        Ok(Self { handle, info })
    }
}

impl<H: Copy, I> ObjectData for BasicObjectData<H, I> {
    type CreateInfo = I;
    type Handle = H;

    fn get_create_info(&self) -> &Self::CreateInfo {
        &self.info
    }
    fn get_handle(&self) -> Self::Handle {
        self.handle
    }
}
