use std::ptr::NonNull;

use pumice::util::result::VulkanResult;

use crate::{inner_context::InnerContext, object::storage::ContextGetStorage};

use self::storage::{ArcHeader, ObjectHeader, ObjectStorage};

pub mod image;
pub mod storage;

pub trait Object: Sized {
    type CreateInfo;
    type Handle;
    type Storage: ObjectStorage<Self>;
    type ImmutableData;
    type MutableData;
    unsafe fn create(
        ctx: &InnerContext,
        info: &Self::CreateInfo,
    ) -> VulkanResult<(Self::Handle, Self::ImmutableData, Self::MutableData)>;
    unsafe fn destroy(ctx: &InnerContext, handle: Self::Handle) -> VulkanResult<()>;
}

pub struct ArcHandle<T: Object>(NonNull<ArcHeader<T>>);

impl<T: Object> ArcHandle<T> {
    pub fn get_header(&self) -> &ObjectHeader<T> {
        unsafe { &self.0.as_ref().header }
    }
}

pub struct CloneMany<T: Object> {
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

            debug_assert!(prev > 0);

            if prev == self.count {
                let storage =
                    <InnerContext as ContextGetStorage<T>>::get_storage((*header).header.ctx);
                T::Storage::destroy(storage.as_ptr(), &mut (*header).header);
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

            debug_assert!(prev > 0);

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

            debug_assert!(prev > 0);

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

            debug_assert!(prev > 0);

            if prev == 1 {
                let storage =
                    <InnerContext as ContextGetStorage<T>>::get_storage((*header).header.ctx);
                T::Storage::destroy(storage.as_ptr(), &mut (*header).header);
            }
        }
    }
}
