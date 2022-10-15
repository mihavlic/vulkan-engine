use std::ptr;

use pumice::{util::result::VulkanResult, vk};
use smallvec::SmallVec;

use crate::{
    context::device::InnerDevice,
    storage::{nostore::NoStore, GetContextStorage},
    synchronization::ReaderWriterState,
    OptionalU32,
};

use super::{ArcHandle, Object};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageCreateInfo {
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

impl ImageCreateInfo {
    pub fn as_raw(&self) -> vk::ImageCreateInfo {
        vk::ImageCreateInfo {
            p_next: ptr::null(),
            flags: self.flags,
            image_type: self.size.as_image_type(),
            format: self.format,
            extent: self.size.as_extent_3d(),
            mip_levels: self.mip_levels,
            array_layers: self.array_layers,
            samples: self.samples,
            tiling: self.tiling,
            usage: self.usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: self.initial_layout,
            ..Default::default()
        }
    }
}

pub struct ImageSynchronizationState {
    owning_family: OptionalU32,
    layout: vk::ImageLayout,
    state: ReaderWriterState,
}

impl ImageSynchronizationState {
    const BLANK: Self = Self {
        owning_family: OptionalU32::NONE,
        layout: vk::ImageLayout::UNDEFINED,
        state: ReaderWriterState::None,
    };
    pub fn with_initial_layout(layout: vk::ImageLayout) -> Self {
        Self {
            layout,
            ..Self::BLANK
        }
    }
}

pub struct Image(pub(crate) ArcHandle<Self>);
impl Object for Image {
    type CreateInfo = ImageCreateInfo;
    type SupplementalInfo = pumice_vma::AllocationCreateInfo;
    type Handle = vk::Image;
    type Storage = NoStore;
    type ObjectData = (pumice_vma::Allocation, ImageSynchronizationState);

    unsafe fn create(
        ctx: &InnerDevice,
        info: &Self::CreateInfo,
        allocation_info: &Self::SupplementalInfo,
    ) -> VulkanResult<(Self::Handle, Self::ObjectData)> {
        let image_info = info.as_raw();
        ctx.allocator
            .create_image(&image_info, allocation_info)
            .map(|(handle, allocation, _)| {
                (
                    handle,
                    (
                        allocation,
                        ImageSynchronizationState::with_initial_layout(info.initial_layout),
                    ),
                )
            })
    }

    unsafe fn destroy(
        ctx: &InnerDevice,
        handle: Self::Handle,
        &(allocation, _): &Self::ObjectData,
    ) -> VulkanResult<()> {
        ctx.allocator.destroy_image(handle, allocation);
        VulkanResult::new_ok(())
    }
}

impl GetContextStorage<Image> for Image {
    fn get_storage(ctx: &InnerDevice) -> &<Image as Object>::Storage {
        &ctx.image_storage
    }
}
