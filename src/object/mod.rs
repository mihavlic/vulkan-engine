macro_rules! create_object {
    ($name:ident) => {
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub(crate) $crate::object::ObjHandle<Self>);

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

        impl std::ops::Deref for $name {
            type Target = $crate::object::ObjRef<$name>;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                // sound because ObjRef is repr(transparent)
                unsafe {
                    std::mem::transmute::<
                        &crate::object::ObjHeader<$name>,
                        &crate::object::ObjRef<$name>,
                    >(self.0 .0.as_ref())
                }
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
use std::{
    borrow::BorrowMut, hash::Hash, mem::ManuallyDrop, ops::Deref, ptr::NonNull,
    sync::atomic::AtomicUsize,
};

use crate::storage::{
    interned::ObjectCreateInfoFingerPrint, MutableShared, ObjectStorage, SynchronizationLock,
};

pub trait ObjectData {
    type CreateInfo;
    type Handle: Copy;
    /// The method must be safe to be called from multiple threads at the same time
    fn get_create_info(&self) -> &Self::CreateInfo;
    /// The method must be safe to be called from multiple threads at the same time
    fn get_handle(&self) -> Self::Handle;
}

pub trait Object: Sized {
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

    fn get_storage(parent: &Self::Parent) -> &Self::Storage;
}

#[repr(C)]
pub struct ObjHeader<T: Object> {
    pub(crate) refcount: AtomicUsize,
    pub(crate) object_data: T::Data,
    pub(crate) storage_data: <T::Storage as ObjectStorage<T>>::StorageData,
    pub(crate) parent: NonNull<T::Parent>,
}

pub struct ObjHandle<T: Object>(pub(crate) NonNull<ObjHeader<T>>);

/// Pointed-to object is atomically reference counted, fields are either immutable or synchronized with MutableShared
unsafe impl<T: Object + Send> Send for ObjHandle<T> {}
unsafe impl<T: Object + Sync> Sync for ObjHandle<T> {}

impl<T: Object> ObjHandle<T> {
    pub unsafe fn get_object_header_ptr(&self) -> *mut ObjHeader<T> {
        self.0.as_ptr()
    }
}

impl<T: Object> ObjRef<T> {
    pub fn get_object_data(&self) -> &T::Data {
        &self.0.object_data
    }
    pub fn get_storage_data(&self) -> &<T::Storage as ObjectStorage<T>>::StorageData {
        &self.0.storage_data
    }
    pub fn get_parent_storage(&self) -> &T::Storage {
        let parent = self.get_parent();
        T::get_storage(parent)
    }
    pub fn get_handle(&self) -> <T::Data as ObjectData>::Handle {
        self.get_object_data().get_handle()
    }
    pub fn get_create_info(&self) -> &<T::Data as ObjectData>::CreateInfo {
        self.get_object_data().get_create_info()
    }
    pub fn get_object_header(&self) -> &ObjHeader<T> {
        &self.0
    }
    pub fn get_parent(&self) -> &T::Parent {
        // safety: dubious
        // user should ensure that parent is the last this to be dropped
        // (or just leak it)
        unsafe { self.get_object_header().parent.as_ref() }
    }
    pub fn lock_storage(&self) -> SynchronizationLock<'_> {
        let storage = self.get_parent_storage();
        storage.acquire_exclusive(self)
    }
    pub fn access_mutable<
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
        let mut refmut = unsafe { mutable.get_mut(&lock) };

        fun2(refmut.borrow_mut())
    }
    /// Duplicate the handle without incrementing the refcount, use this only when holding the
    /// the original handle.
    pub unsafe fn make_weak_copy(&self) -> ManuallyDrop<ObjHandle<T>> {
        ManuallyDrop::new(ObjHandle(NonNull::from(&self.0)))
    }
}

impl<T: Object> PartialEq for ObjHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T: Object> PartialOrd for ObjHandle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl<T: Object> Eq for ObjHandle<T> {}
impl<T: Object> Ord for ObjHandle<T> {
    fn cmp(&self, _other: &Self) -> std::cmp::Ordering {
        todo!()
    }
}
impl<T: Object> Hash for ObjHandle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

pub(crate) struct CloneMany<T: Object> {
    handle: ObjHandle<T>,
    count: usize,
}

impl<T: Object> Iterator for CloneMany<T> {
    type Item = ObjHandle<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.count > 0 {
            self.count -= 1;
            // safety: we've bulk incremented the refcount at the creation of CloneMany
            // so this new handle is accounted for
            let copy = unsafe { self.handle.make_weak_copy() };
            return Some(ManuallyDrop::into_inner(copy));
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
                .get_object_header()
                .refcount
                .fetch_sub(self.count, std::sync::atomic::Ordering::SeqCst);

            assert!(prev > 0);

            // if we just subtracted the same value as was in self.count, self.count is now 0, destroy the object
            if prev == self.count {
                let storage = self.handle.get_parent_storage();
                T::Storage::destroy(storage, self.handle.make_weak_copy()).unwrap();
            }
        }
    }
}

impl<T: Object> ObjHandle<T> {
    pub(crate) fn clone_many<'a>(&'a self, count: usize) -> CloneMany<T> {
        unsafe {
            let header = self.0;
            let prev = (*header.as_ptr())
                .refcount
                .fetch_add(count, std::sync::atomic::Ordering::SeqCst);

            assert!(prev > 0);

            CloneMany {
                handle: ObjHandle(self.0),
                count,
            }
        }
    }
}

impl<T: Object> Clone for ObjHandle<T> {
    fn clone(&self) -> Self {
        unsafe {
            let header = self.0;
            let prev = (*header.as_ptr())
                .refcount
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            assert!(prev > 0);

            ObjHandle(header)
        }
    }
}

impl<T: Object> Drop for ObjHandle<T> {
    fn drop(&mut self) {
        let prev = self
            .get_object_header()
            .refcount
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

        assert!(prev > 0);

        if prev == 1 {
            let storage = self.get_parent_storage();
            unsafe {
                T::Storage::destroy(storage, self.make_weak_copy()).unwrap();
            }
        }
    }
}

pub struct BasicObjectData<H, I> {
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

impl<I: ObjectCreateInfoFingerPrint, H> ObjectCreateInfoFingerPrint for BasicObjectData<H, I> {
    fn get_fingerprint(&self) -> u128 {
        self.info.get_fingerprint()
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

#[repr(transparent)]
pub struct ObjRef<T: Object>(ObjHeader<T>);

impl<T: Object> std::ops::Deref for ObjHandle<T> {
    type Target = ObjRef<T>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { std::mem::transmute::<&ObjHeader<T>, &ObjRef<T>>(self.0.as_ref()) }
    }
}
