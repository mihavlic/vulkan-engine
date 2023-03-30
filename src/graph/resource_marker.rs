use pumice::vk;
use std::ops::{Deref, DerefMut};

use super::{
    compile::CombinedResourceHandle,
    record::{PassBufferData, PassImageData},
    RawHandle,
};

pub trait TypeOption<T>: From<T> {
    fn new_some(val: T) -> Self;
    fn new_none() -> Self;
    fn get(&self) -> &T;
    fn get_mut(&mut self) -> &mut T;
    fn unwrap(self) -> T;
    fn to_option(self) -> Option<T>;

    type This<A>: TypeOption<A>;
    fn as_ref(&self) -> Self::This<&T>;
    fn as_mut(&mut self) -> Self::This<&mut T>;
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct TypeSome<T>(T);
impl<T> TypeOption<T> for TypeSome<T> {
    #[inline(always)]
    fn new_some(val: T) -> Self {
        TypeSome(val)
    }

    #[inline(always)]
    fn new_none() -> Self {
        unreachable!()
    }

    #[inline(always)]
    fn get(&self) -> &T {
        &self.0
    }

    #[inline(always)]
    fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }

    #[inline(always)]
    fn unwrap(self) -> T {
        self.0
    }

    #[inline(always)]
    fn to_option(self) -> Option<T> {
        Some(self.0)
    }
    type This<A> = TypeSome<A>;

    #[inline(always)]
    fn as_ref(&self) -> Self::This<&T> {
        TypeSome(&self.0)
    }

    #[inline(always)]
    fn as_mut(&mut self) -> Self::This<&mut T> {
        TypeSome(&mut self.0)
    }
}
impl<T: Clone> Clone for TypeSome<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T: Copy> Copy for TypeSome<T> {}
impl<T> From<T> for TypeSome<T> {
    #[inline(always)]
    fn from(value: T) -> Self {
        TypeSome::new_some(value)
    }
}

pub struct TypeNone<T>(std::marker::PhantomData<fn() -> T>);
impl<T> TypeOption<T> for TypeNone<T> {
    #[inline(always)]
    fn new_some(_val: T) -> Self {
        TypeNone(std::marker::PhantomData)
    }

    #[inline(always)]
    fn new_none() -> Self {
        TypeNone(std::marker::PhantomData)
    }

    #[inline(always)]
    fn get(&self) -> &T {
        unreachable!()
    }

    #[inline(always)]
    fn get_mut(&mut self) -> &mut T {
        unreachable!()
    }

    #[inline(always)]
    fn unwrap(self) -> T {
        unreachable!()
    }

    #[inline(always)]
    fn to_option(self) -> Option<T> {
        None
    }
    type This<A> = TypeNone<A>;

    #[inline(always)]
    fn as_ref(&self) -> Self::This<&T> {
        TypeNone(std::marker::PhantomData)
    }

    #[inline(always)]
    fn as_mut(&mut self) -> Self::This<&mut T> {
        TypeNone(std::marker::PhantomData)
    }
}
impl<T> Clone for TypeNone<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(std::marker::PhantomData)
    }
}
impl<T> Copy for TypeNone<T> {}
impl<T> From<T> for TypeNone<T> {
    #[inline(always)]
    fn from(value: T) -> Self {
        TypeNone::new_some(value)
    }
}
// #[derive(PartialEq, Eq, PartialOrd, Ord)]
impl<T> PartialEq for TypeNone<T> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl<T> Eq for TypeNone<T> {}
impl<T> PartialOrd for TypeNone<T> {
    fn partial_cmp(&self, _other: &Self) -> Option<std::cmp::Ordering> {
        Some(std::cmp::Ordering::Equal)
    }
}
impl<T> Ord for TypeNone<T> {
    fn cmp(&self, _other: &Self) -> std::cmp::Ordering {
        std::cmp::Ordering::Equal
    }
}

pub trait ResourceData {
    fn access(&self) -> vk::AccessFlags2KHR;
    fn stages(&self) -> vk::PipelineStageFlags2KHR;
    fn start_layout(&self) -> vk::ImageLayout;
    fn raw_resource_handle(&self) -> RawHandle;
    fn graph_resource(&self) -> CombinedResourceHandle;
    fn end_layout(&self) -> Option<vk::ImageLayout>;
}

impl ResourceData for PassImageData {
    #[inline(always)]
    fn access(&self) -> vk::AccessFlags2KHR {
        self.access
    }

    #[inline(always)]
    fn stages(&self) -> vk::PipelineStageFlags2KHR {
        self.stages
    }

    #[inline(always)]
    fn start_layout(&self) -> vk::ImageLayout {
        self.start_layout
    }

    #[inline(always)]
    fn raw_resource_handle(&self) -> RawHandle {
        self.handle.to_raw()
    }

    #[inline(always)]
    fn graph_resource(&self) -> CombinedResourceHandle {
        CombinedResourceHandle::new_image(self.handle)
    }

    #[inline(always)]
    fn end_layout(&self) -> Option<vk::ImageLayout> {
        self.end_layout
    }
}

impl ResourceData for PassBufferData {
    #[inline(always)]
    fn access(&self) -> vk::AccessFlags2KHR {
        self.access
    }

    #[inline(always)]
    fn stages(&self) -> vk::PipelineStageFlags2KHR {
        self.stages
    }

    #[inline(always)]
    fn start_layout(&self) -> vk::ImageLayout {
        unreachable!("This code path should never be taken!")
    }

    #[inline(always)]
    fn raw_resource_handle(&self) -> RawHandle {
        self.handle.to_raw()
    }

    #[inline(always)]
    fn graph_resource(&self) -> CombinedResourceHandle {
        CombinedResourceHandle::new_buffer(self.handle)
    }

    #[inline(always)]
    fn end_layout(&self) -> Option<vk::ImageLayout> {
        unreachable!("This code path should never be taken!")
    }
}

pub trait ResourceMarker {
    const IS_IMAGE: bool;
    const IS_BUFFER: bool = !Self::IS_IMAGE;

    type IfImage<T>: TypeOption<T>;
    type IfBuffer<T>: TypeOption<T>;
    fn when_image<T, F: FnOnce() -> T>(fun: F) -> Self::IfImage<T>;
    fn when_buffer<T, F: FnOnce() -> T>(fun: F) -> Self::IfBuffer<T>;
    type Data: ResourceData;

    type EitherOut<Image, Buffer>;
    fn select<Image, Buffer>(
        a: Self::IfImage<Image>,
        b: Self::IfBuffer<Buffer>,
    ) -> Self::EitherOut<Image, Buffer>;
    fn select_ref<'a, Image, Buffer>(
        a: &'a Self::IfImage<Image>,
        b: &'a Self::IfBuffer<Buffer>,
    ) -> &'a Self::EitherOut<Image, Buffer>;
    fn select_ref_mut<'a, Image, Buffer>(
        a: &'a mut Self::IfImage<Image>,
        b: &'a mut Self::IfBuffer<Buffer>,
    ) -> &'a mut Self::EitherOut<Image, Buffer>;
    fn select_with<Image, Buffer, FImage: FnOnce() -> Image, FBuffer: FnOnce() -> Buffer>(
        image: FImage,
        buffer: FBuffer,
    ) -> Self::EitherOut<Image, Buffer>;

    fn new_either<Image, Buffer>(
        value: Self::EitherOut<Image, Buffer>,
    ) -> TypeEither<Self, Image, Buffer>;
}

#[derive(Clone, Default)]
pub struct ImageMarker;
impl ResourceMarker for ImageMarker {
    const IS_IMAGE: bool = true;

    type IfImage<T> = TypeSome<T>;
    type IfBuffer<T> = TypeNone<T>;
    type Data = PassImageData;

    #[inline(always)]
    fn when_image<T, F: FnOnce() -> T>(fun: F) -> Self::IfImage<T> {
        TypeSome(fun())
    }

    #[inline(always)]
    fn when_buffer<T, F: FnOnce() -> T>(_fun: F) -> Self::IfBuffer<T> {
        TypeNone(std::marker::PhantomData)
    }

    type EitherOut<Image, Buffer> = Image;

    #[inline(always)]
    fn select<Image, Buffer>(
        a: Self::IfImage<Image>,
        _b: Self::IfBuffer<Buffer>,
    ) -> Self::EitherOut<Image, Buffer> {
        a.unwrap()
    }

    #[inline(always)]
    fn select_ref<'a, Image, Buffer>(
        a: &'a Self::IfImage<Image>,
        _b: &'a Self::IfBuffer<Buffer>,
    ) -> &'a Self::EitherOut<Image, Buffer> {
        a.get()
    }

    #[inline(always)]
    fn select_ref_mut<'a, Image, Buffer>(
        a: &'a mut Self::IfImage<Image>,
        _b: &'a mut Self::IfBuffer<Buffer>,
    ) -> &'a mut Self::EitherOut<Image, Buffer> {
        a.get_mut()
    }

    #[inline(always)]
    fn new_either<Image, Buffer>(
        value: Self::EitherOut<Image, Buffer>,
    ) -> TypeEither<Self, Image, Buffer> {
        TypeEither(Self::IfImage::new_some(value), Self::IfBuffer::new_none())
    }

    #[inline(always)]
    fn select_with<Image, Buffer, FImage: FnOnce() -> Image, FBuffer: FnOnce() -> Buffer>(
        image: FImage,
        _buffer: FBuffer,
    ) -> Self::EitherOut<Image, Buffer> {
        image()
    }
}

#[derive(Clone, Default)]
pub struct BufferMarker;
impl ResourceMarker for BufferMarker {
    const IS_IMAGE: bool = false;

    type IfImage<T> = TypeNone<T>;
    type IfBuffer<T> = TypeSome<T>;
    type Data = PassBufferData;

    #[inline(always)]
    fn when_image<T, F: FnOnce() -> T>(_fun: F) -> Self::IfImage<T> {
        TypeNone(std::marker::PhantomData)
    }

    #[inline(always)]
    fn when_buffer<T, F: FnOnce() -> T>(fun: F) -> Self::IfBuffer<T> {
        TypeSome(fun())
    }

    type EitherOut<Image, Buffer> = Buffer;

    #[inline(always)]
    fn select<Image, Buffer>(
        _a: Self::IfImage<Image>,
        b: Self::IfBuffer<Buffer>,
    ) -> Self::EitherOut<Image, Buffer> {
        b.unwrap()
    }

    #[inline(always)]
    fn select_ref<'a, Image, Buffer>(
        _a: &'a Self::IfImage<Image>,
        b: &'a Self::IfBuffer<Buffer>,
    ) -> &'a Self::EitherOut<Image, Buffer> {
        b.get()
    }

    #[inline(always)]
    fn select_ref_mut<'a, Image, Buffer>(
        _a: &'a mut Self::IfImage<Image>,
        b: &'a mut Self::IfBuffer<Buffer>,
    ) -> &'a mut Self::EitherOut<Image, Buffer> {
        b.get_mut()
    }

    #[inline(always)]
    fn new_either<Image, Buffer>(
        value: Self::EitherOut<Image, Buffer>,
    ) -> TypeEither<Self, Image, Buffer> {
        TypeEither(Self::IfImage::new_none(), Self::IfBuffer::new_some(value))
    }

    #[inline(always)]
    fn select_with<Image, Buffer, FImage: FnOnce() -> Image, FBuffer: FnOnce() -> Buffer>(
        _image: FImage,
        buffer: FBuffer,
    ) -> Self::EitherOut<Image, Buffer> {
        buffer()
    }
}

#[derive(Clone, Default)]
pub struct TypeEither<T: ResourceMarker + ?Sized, Image, Buffer>(
    T::IfImage<Image>,
    T::IfBuffer<Buffer>,
);

impl<T: ResourceMarker, IfImage, IfBuffer> TypeEither<T, IfImage, IfBuffer> {
    #[inline(always)]
    pub fn new(value: T::EitherOut<IfImage, IfBuffer>) -> Self {
        T::new_either(value)
    }
    #[inline(always)]
    pub fn decompose(self) -> T::EitherOut<IfImage, IfBuffer> {
        T::select(self.0, self.1)
    }
    #[inline(always)]
    pub fn map<TI, TB, FImage: FnOnce(IfImage) -> TI, FBuffer: FnOnce(IfBuffer) -> TB>(
        self,
        map_image: FImage,
        map_buffer: FBuffer,
    ) -> TypeEither<T, TI, TB> {
        T::new_either(T::select_with(
            || map_image(self.0.unwrap()),
            || map_buffer(self.1.unwrap()),
        ))
    }
}

impl<T: ResourceMarker, IfImage, IfBuffer> Deref for TypeEither<T, IfImage, IfBuffer> {
    type Target = T::EitherOut<IfImage, IfBuffer>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        T::select_ref(&self.0, &self.1)
    }
}
impl<T: ResourceMarker, IfImage, IfBuffer> DerefMut for TypeEither<T, IfImage, IfBuffer> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        T::select_ref_mut(&mut self.0, &mut self.1)
    }
}
