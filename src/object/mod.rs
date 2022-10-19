mod buffer;
mod image;

pub use buffer::*;
pub use image::*;

use pumice::util::result::VulkanResult;
use std::ptr::NonNull;

use crate::{
    context::device::InnerDevice,
    storage::{ArcHeader, ObjectHeader, ObjectStorage},
};

pub(crate) trait Object: Sized {
    type CreateInfo;
    type SupplementalInfo;
    type Handle;
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
        data: &Self::ObjectData,
    ) -> VulkanResult<()>;

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage;
}

pub(crate) struct ArcHandle<T: Object>(pub(crate) NonNull<ArcHeader<T>>);

impl<T: Object> ArcHandle<T> {
    pub fn get_header(&self) -> &ObjectHeader<T> {
        unsafe { &self.0.as_ref().header }
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
        unsafe {
            let header = self.handle.0.as_ptr();
            let prev = (*header)
                .refcount
                .fetch_sub(self.count, std::sync::atomic::Ordering::SeqCst);

            assert!(prev > 0);

            if prev == self.count {
                let storage = T::get_storage((*header).header.parent());
                T::Storage::destroy(storage, header);
            }
        }
    }
}

impl<T: Object> ArcHandle<T> {
    fn clone_many<'a>(&'a self, count: usize) -> CloneMany<T> {
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
            let header = self.0.as_ptr();
            let prev = (*header)
                .refcount
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

            assert!(prev > 0);

            if prev == 1 {
                let mut storage = T::get_storage((*header).header.parent());
                T::Storage::destroy(storage, header);
            }
        }
    }
}
