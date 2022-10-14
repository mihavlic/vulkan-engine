use std::{
    alloc::Layout,
    borrow::Borrow,
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::atomic::AtomicUsize,
};

use pumice::util::result::VulkanResult;

use crate::inner_context::InnerContext;

use super::{ArcHandle, Object};

pub trait ObjectStorage<T: Object>: Sized {
    type StorageData;
    // unsafe fn get_or_create_with<F: FnOnce(&T::CreateInfo) -> VulkanResult<T::Handle>>(
    //     this: *mut Self,
    //     with: F,
    //     info: T::CreateInfo,
    //     ctx: *mut InnerContext,
    // ) -> VulkanResult<ArcHandle<T>>;
    unsafe fn get_or_create(
        this: *mut Self,
        info: T::CreateInfo,
        ctx: *mut InnerContext,
    ) -> VulkanResult<ArcHandle<T>>;
    unsafe fn destroy(this: *mut Self, header: *mut ObjectHeader<T>);
}

pub struct ObjectHeader<T: Object> {
    pub(crate) handle: T::Handle,
    pub(crate) info: T::CreateInfo,
    pub(crate) storage_data: <T::Storage as ObjectStorage<T>>::StorageData,
    pub(crate) immutable_data: <T as Object>::ImmutableData,
    pub(crate) mutable_data: MutableShared<<T as Object>::MutableData>,
    pub(crate) ctx: NonNull<InnerContext>,
}

#[repr(C)]
pub struct ArcHeader<T: Object> {
    pub(crate) refcount: AtomicUsize,
    pub(crate) header: ObjectHeader<T>,
}

impl<T: Object> ContextGetStorage<T> for InnerContext {
    fn get_storage(ctx: NonNull<InnerContext>) -> NonNull<T::Storage> {
        todo!()
    }
}

pub trait ContextGetStorage<T: Object> {
    fn get_storage(ctx: NonNull<InnerContext>) -> NonNull<T::Storage>;
}

// when we have guaranteed access from a single thread

pub trait SynchronizationGuarantee {}

pub struct ThreadLocal(());

impl ThreadLocal {
    pub unsafe fn pledge_single_threaded() -> Self {
        Self(())
    }
}

impl SynchronizationGuarantee for ThreadLocal {}

// mutex

impl<'a> SynchronizationGuarantee for std::sync::MutexGuard<'a, ()> {}

pub struct MutableShared<T>(UnsafeCell<T>);

impl<T> MutableShared<T> {
    pub(crate) fn new(whalue: T) -> Self {
        MutableShared(UnsafeCell::new(whalue))
    }
    pub unsafe fn borrow_mut<'a>(
        &'a self,
        key: &'a (impl SynchronizationGuarantee + 'a),
    ) -> MutableSharedRef<'a, T> {
        MutableSharedRef(&self)
    }
}

pub struct MutableSharedRef<'a, T>(&'a MutableShared<T>);

impl<'a, T> Deref for MutableSharedRef<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        // we've already used unsafe to make the MutableSharedRef
        unsafe { &*UnsafeCell::get(&self.0 .0) }
    }
}

impl<'a, T> DerefMut for MutableSharedRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // we've already used unsafe to make the MutableSharedRef
        unsafe { &mut *UnsafeCell::get(&self.0 .0) }
    }
}

// storage methods

pub struct NoStore;

impl<T: Object<Storage = Self>> ObjectStorage<T> for NoStore {
    type StorageData = ();

    // unsafe fn get_or_create_with<
    //     F: FnOnce(
    //         &<T as Object>::CreateInfo,
    //     ) -> pumice::util::result::VulkanResult<<T as Object>::Handle>,
    // >(
    //     this: *mut Self,
    //     with: F,
    //     info: <T as Object>::CreateInfo,
    //     ctx: *mut InnerContext,
    // ) -> pumice::util::result::VulkanResult<ArcHandle<T>> {
    //     with(&info).map(|handle| {
    //         let header = create_header(handle, info, (), ctx);
    //         let alloc = Box::new(ArcHeader {
    //             refcount: AtomicUsize::new(1),
    //             header,
    //         });
    //         let ptr = Box::leak(alloc);
    //         ArcHandle(NonNull::new_unchecked(ptr))
    //     })
    // }

    unsafe fn destroy(s: *mut Self, header: *mut ObjectHeader<T>) {
        let alloc = Box::from_raw(header);
        T::destroy(alloc.ctx.as_ref(), alloc.handle).unwrap();
        std::alloc::dealloc(header as *mut _, Layout::new::<ObjectHeader<T>>())
    }

    unsafe fn get_or_create(
        this: *mut Self,
        info: <T as Object>::CreateInfo,
        ctx: *mut InnerContext,
    ) -> VulkanResult<ArcHandle<T>> {
        todo!()
    }
}

unsafe fn create_header<T: Object>(
    info: T::CreateInfo,
    data: <T::Storage as ObjectStorage<T>>::StorageData,
    ctx: *mut InnerContext,
) -> VulkanResult<ObjectHeader<T>> {
    T::create(&*ctx, &info).map(|(handle, immutable, mutable)| ObjectHeader {
        handle,
        info,
        storage_data: data,
        immutable_data: immutable,
        mutable_data: MutableShared::new(mutable),
        ctx: NonNull::new(ctx).unwrap(),
    })
}
