use std::ptr;

use pumice::{util::result::VulkanResult, vk};
use smallvec::SmallVec;

use crate::inner_context::InnerContext;

use super::{storage::NoStore, ArcHandle, Object};

#[derive(Clone, Copy)]
pub enum Extent {
    D1(u32),
    D2(u32, u32),
    D3(u32, u32, u32),
}

impl Extent {
    pub fn as_image_type(&self) -> vk::ImageType {
        match self {
            Extent::D1(_) => vk::ImageType::T1D,
            Extent::D2(_, _) => vk::ImageType::T2D,
            Extent::D3(_, _, _) => vk::ImageType::T3D,
        }
    }
    pub fn as_extent_2d(&self) -> vk::Extent2D {
        match *self {
            Extent::D1(w) => vk::Extent2D {
                width: w,
                height: 1,
            },
            Extent::D2(w, h) => vk::Extent2D {
                width: w,
                height: h,
            },
            Extent::D3(w, h, _) => vk::Extent2D {
                width: w,
                height: h,
            },
        }
    }
    pub fn as_extent_3d(&self) -> vk::Extent3D {
        match *self {
            Extent::D1(w) => vk::Extent3D {
                width: w,
                height: 1,
                depth: 1,
            },
            Extent::D2(w, h) => vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            },
            Extent::D3(w, h, d) => vk::Extent3D {
                width: w,
                height: h,
                depth: d,
            },
        }
    }
}

pub struct ImageInfo {
    pub flags: vk::ImageCreateFlags,
    pub size: Extent,
    pub format: vk::Format,
    pub samples: vk::SampleCountFlags,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub initial_layout: vk::ImageLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OptionalU32(u32);

impl OptionalU32 {
    pub const NONE: Self = Self(u32::MAX);

    pub fn new(value: u32) -> Self {
        assert!(value != u32::MAX);
        Self(value)
    }
    pub fn new_none() -> Self {
        Self::NONE
    }
    pub fn get(&self) -> Option<u32> {
        if *self == Self::NONE {
            None
        } else {
            Some(self.0)
        }
    }
    pub fn set(&mut self, value: Option<u32>) {
        if let Some(value) = value {
            assert!(value != u32::MAX);
            *self = Self(value);
        } else {
            *self = Self::NONE;
        }
    }
}

pub struct CommandBufferSubmission(u32);

pub enum ReaderWriterState {
    Read(SmallVec<[CommandBufferSubmission; 4]>),
    Write(CommandBufferSubmission),
}

pub struct ImageSynchronizationState {
    owning_family: u32,
    layout: vk::ImageLayout,
    state: ReaderWriterState,
}

pub struct Image(ArcHandle<Self>);
impl Object for Image {
    type CreateInfo = ImageInfo;
    type Handle = vk::Image;
    type Storage = NoStore;
    type ImmutableData = ();
    type MutableData = ImageSynchronizationState;

    unsafe fn create(
        ctx: &InnerContext,
        info: &Self::CreateInfo,
    ) -> VulkanResult<(Self::Handle, Self::ImmutableData, Self::MutableData)> {
        let create_info = vk::ImageCreateInfo {
            p_next: ptr::null(),
            flags: info.flags,
            image_type: info.size.as_image_type(),
            format: info.format,
            extent: info.size.as_extent_3d(),
            mip_levels: info.mip_levels,
            array_layers: info.array_layers,
            samples: info.samples,
            tiling: info.tiling,
            usage: info.usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: info.initial_layout,
            ..Default::default()
        };

        // ctx.create_image(&create_info, ctx.allocator_callbacks())
        todo!()
    }

    unsafe fn destroy(ctx: &InnerContext, handle: Self::Handle) -> VulkanResult<()> {
        // ctx.destroy_image(image, allocator)
        todo!()
    }
}
