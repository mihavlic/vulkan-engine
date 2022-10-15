use std::ptr::NonNull;

use pumice::util::result::VulkanResult;

use super::{create_header, ArcHeader, ObjectStorage, ReentrantMutex};
use crate::{
    context::device::InnerDevice,
    object::{ArcHandle, Object},
};

pub struct NoStore {
    lock: ReentrantMutex,
}

impl NoStore {
    pub fn new() -> Self {
        NoStore {
            lock: ReentrantMutex::new(),
        }
    }
}

impl<T: Object<Storage = Self>> ObjectStorage<T> for NoStore {
    type StorageData = ();

    unsafe fn get_or_create(
        &self,
        info: <T as Object>::CreateInfo,
        supplemental: <T as Object>::SupplementalInfo,
        ctx: &InnerDevice,
    ) -> VulkanResult<ArcHandle<T>> {
        self.lock.with_locked(|| {
            create_header(ctx, info, supplemental, ()).map(|header| {
                let boxed = Box::new(ArcHeader {
                    refcount: 1.into(),
                    header,
                });
                let leak = Box::leak(boxed);
                ArcHandle(NonNull::from(leak))
            })
        })
    }

    unsafe fn destroy(&self, header: *mut ArcHeader<T>) {
        let alloc = Box::from_raw(header);
        let ArcHeader { refcount, header } = *alloc;

        self.lock
            .with_locked(|| T::destroy(header.ctx(), header.handle, &header.object_data).unwrap());
    }
}
