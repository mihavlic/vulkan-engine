#![allow(unused)]

use std::{marker::PhantomData, mem::ManuallyDrop, ptr::NonNull, sync::atomic::AtomicUsize};

use slag::{
    deep_copy::{DeepCopy, DeepCopyBox},
    dumb_hash::DumbHash,
    util::result::VulkanResult,
    vk,
};

pub struct InnerContext {
    device: slag::DeviceWrapper,
}

#[repr(C)]
pub struct ObjectHeader<T: Object, S: ObjectStorage<T>> {
    refcount: AtomicUsize,
    handle: T::Handle,
    ctx: NonNull<InnerContext>,
    storage_data: S::StorageData,
    info: DeepCopyBox<T::CreateInfo>,
}

pub struct ArcHandle<T: Object, S: ObjectStorage<T>>(NonNull<ObjectHeader<T, S>>);

impl<T: Object, S: ObjectStorage<T>> Clone for ArcHandle<T, S> {
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

impl<T: Object, S: ObjectStorage<T>> Drop for ArcHandle<T, S> {
    fn drop(&mut self) {
        unsafe {
            let header = self.0;
            let prev = (*header.as_ptr())
                .refcount
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            debug_assert!(prev > 0);
            if prev == 0 {
                let storage =
                    <InnerContext as ContextGetStorage<T, S>>::get_storage((*header.as_ptr()).ctx);
                S::destroy(header);
            }
        }
    }
}

impl<T: Object, S: ObjectStorage<T>> ContextGetStorage<T, S> for InnerContext {
    fn get_storage(ctx: NonNull<InnerContext>) -> NonNull<S> {
        todo!()
    }
}

pub trait Object: Sized {
    type CreateInfo: DeepCopy;
    type Handle;
    type Storage: ObjectStorage<Self>;
    fn create(device: &slag::DeviceWrapper, info: &Self::CreateInfo) -> VulkanResult<Self::Handle>;
    fn destroy(device: &slag::DeviceWrapper, handle: Self::Handle) -> VulkanResult<()>;
}

pub trait ObjectStorage<T: Object>: Sized {
    type StorageData;
    fn get_or_create(
        ctx: *mut InnerContext,
        info: &T::CreateInfo,
    ) -> VulkanResult<ArcHandle<T, Self>>;
    fn destroy(header: NonNull<ObjectHeader<T, Self>>);
    fn create_header(
        ctx: NonNull<InnerContext>,
        info: &T::CreateInfo,
        data: Self::StorageData,
    ) -> VulkanResult<ObjectHeader<T, Self>> {
        T::create(&(*ctx.as_ptr()).device, info).map(|handle| {
            let copy = info.deep_copy();
            ObjectHeader {
                refcount: AtomicUsize::new(1),
                handle: handle,
                ctx: ctx,
                storage_data: data,
                info: copy,
            }
        })
    }
}

pub trait ContextGetStorage<T: Object, S: ObjectStorage<T>> {
    fn get_storage(ctx: NonNull<InnerContext>) -> NonNull<S>;
}

macro_rules! arc_handle_impl {
    ($name:ident: $storage:ident [$object:ident] {
        create($device:ident, $info:ident : $info_ty:path) $create_code:expr,
        destroy($device2:ident, $handle:ident : $handle_ty:path) $destroy_code:expr
    }) => {
        pub struct $name(ArcHandle);
        impl VkObject for $name {
            type CreateInfo = $info_ty;
            type Handle = $handle_ty;
            fn create(
                $device: slag::DeviceWrapper,
                $info: &Self::CreateInfo,
            ) -> VulkanResult<Self::Handle> {
                $create_code
            }
            fn destroy($device2: slaf::DeviceWrapper, $handle: Self::Handle) -> VulkanResult<()> {
                $destroy_code
            }
        }
    };
}

// arc_handle_impl!(
//     Image:
// );

struct NoStore;

impl<T: Object> ObjectStorage<T> for NoStore {
    type StorageData = Box<ObjectHeader<T, Self>>;
    fn get_or_create(
        ctx: *mut InnerContext,
        info: &<T as Object>::CreateInfo,
    ) -> VulkanResult<ArcHandle<T, Self>> {
        todo!()
    }
    fn destroy(header: *mut ObjectHeader<T, Self>) {
        unsafe {
            T::destroy(&(*(*header).ctx).device, (*header).handle).unwrap();
            std::ptr::drop_in_place(header);
        }
    }
}

// struct Context();

// impl Context {
//     fn create_image(&mut self) -> Image {
//         todo!()
//     }
//     fn create_buffer(&mut self) -> Buffer {
//         todo!()
//     }
//     fn create_graphics_pipeline(&mut self, info: CreateGraphicsPipeline) -> GraphicsPipeline {
//         todo!()
//     }
//     fn create_sampler(&mut self, info: &vk::SamplerCreateInfo) -> Sampler {
//         todo!()
//     }
//     fn create_queue(&mut self) -> Queue {
//         todo!()
//     }
//     fn create_event(&mut self) -> Event {
//         todo!()
//     }
//     fn create_fence(&mut self) -> Fence {
//         todo!()
//     }
//     fn create_semaphore(&mut self) -> Sampler {
//         todo!()
//     }
//     fn create_descriptor_layout(&mut self) -> DescriptorLayout {
//         todo!()
//     }
// }

// pub struct CreateGraphicsPipeline {}

// macro_rules! handle_impl {
//     ($($name:ident),+) => {
//         $(
//             pub struct $name(u32);
//             impl $name {
//                 pub fn new(len: usize) -> Self {
//                     Self(len.try_into().unwrap())
//                 }
//                 pub fn raw(&self) -> usize {
//                     self.0 as usize
//                 }
//             }
//         )+
//     };
// }

// handle_impl!(
//     Image,
//     Buffer,
//     GraphicsPipeline,
//     Sampler,
//     Queue,
//     Event,
//     Fence,
//     DescriptorLayout
// );

// fn main() {}
