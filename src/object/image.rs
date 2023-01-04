use std::{
    borrow::BorrowMut,
    cell::UnsafeCell,
    hash::{BuildHasher, Hash, Hasher},
    ops::DerefMut,
    ptr,
};

use pumice::{vk, VulkanResult};
use smallvec::SmallVec;

use crate::{
    arena::uint::OptionalU32,
    batch::GenerationId,
    context::device::Device,
    storage::{constant_ahash_hasher, nostore::SimpleStorage, MutableShared, ObjectStorage},
    submission::ReaderWriterState,
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
    pub sharing_mode_concurrent: bool,
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
            sharing_mode: vk::SharingMode(self.sharing_mode_concurrent as _),
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

#[derive(Clone, Hash)]
pub struct ImageViewCreateInfo {
    view_type: vk::ImageViewType,
    format: vk::Format,
    components: vk::ComponentMapping,
    subresource_range: vk::ImageSubresourceRange,
}

impl ImageViewCreateInfo {
    fn get_hash(&self) -> u32 {
        let mut hasher = constant_ahash_hasher();
        self.hash(&mut hasher);

        // this is dumb but with a high quality function truncating like this should be kind of ok
        hasher.finish() as u32
    }
}

struct ImageViewEntry {
    handle: vk::ImageView,
    info_hash: u32,
    last_use: GenerationId,
}

pub struct ImageMutableState {
    views: SmallVec<[ImageViewEntry; 2]>,
    synchronization: ImageSynchronizationState,
}

impl ImageMutableState {
    pub fn with_initial_layout(layout: vk::ImageLayout) -> Self {
        Self {
            views: SmallVec::new(),
            synchronization: ImageSynchronizationState::with_initial_layout(layout),
        }
    }
    pub unsafe fn get_view(
        &mut self,
        self_handle: vk::Image,
        info: &ImageViewCreateInfo,
        batch_id: GenerationId,
        device: &Device,
    ) -> VulkanResult<vk::ImageView> {
        let hash = info.get_hash();

        if let Some(found) = self.views.iter_mut().find(|v| v.info_hash == hash) {
            found.last_use = batch_id;
            VulkanResult::Ok(found.handle)
        } else {
            let raw = vk::ImageViewCreateInfo {
                image: self_handle,
                view_type: info.view_type,
                format: info.format,
                components: info.components.clone(),
                subresource_range: info.subresource_range.clone(),
                ..Default::default()
            };

            let view = device
                .device()
                .create_image_view(&raw, device.allocator_callbacks())?;

            let entry = ImageViewEntry {
                handle: view,
                info_hash: hash,
                last_use: batch_id,
            };

            self.views.push(entry);

            VulkanResult::Ok(view)
        }
    }
    pub unsafe fn destroy(self, device: &Device) {
        for view in self.views {
            device
                .device()
                .destroy_image_view(view.handle, device.allocator_callbacks());
        }
    }
}

pub struct Image(pub(crate) ArcHandle<Self>);
impl Object for Image {
    type CreateInfo = ImageCreateInfo;
    type SupplementalInfo = pumice_vma::AllocationCreateInfo;
    type Handle = vk::Image;
    type Storage = SimpleStorage<Self>;
    type ObjectData = (pumice_vma::Allocation, MutableShared<ImageMutableState>);

    type Parent = Device;

    unsafe fn create(
        ctx: &Device,
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
                        MutableShared::new(ImageMutableState::with_initial_layout(
                            info.initial_layout,
                        )),
                    ),
                )
            })
    }

    unsafe fn destroy(
        ctx: &Device,
        handle: Self::Handle,
        &(allocation, _): &Self::ObjectData,
    ) -> VulkanResult<()> {
        ctx.allocator.destroy_image(handle, allocation);
        VulkanResult::Ok(())
    }

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.image_storage
    }
}

impl Image {
    unsafe fn get_view(
        &self,
        info: &ImageViewCreateInfo,
        batch_id: GenerationId,
    ) -> VulkanResult<vk::ImageView> {
        let storage = self.0.get_storage();
        let header = storage.read_object(&self.0);
        let mut data = header.object_data.1.borrow_mut(header.get_lock());
        data.get_view(header.handle, info, batch_id, self.0.get_parent())
    }
}
