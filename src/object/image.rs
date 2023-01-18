use std::{
    borrow::BorrowMut,
    cell::UnsafeCell,
    hash::{BuildHasher, Hash, Hasher},
    ops::DerefMut,
    ptr,
};

use pumice::{vk, VulkanResult};
use smallvec::{smallvec, SmallVec};

use crate::{
    arena::uint::OptionalU32,
    graph::resource_marker::{BufferMarker, ImageMarker, ResourceMarker, TypeOption},
    device::{batch::GenerationId, submission::QueueSubmission, Device},
    storage::{
        constant_ahash_hasher, nostore::SimpleStorage, MutableShared, ObjectHeader, ObjectStorage,
        SynchronizationLock,
    },
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
    pub fn to_vk(&self) -> vk::ImageCreateInfo {
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

pub enum SynchronizeResult {
    None,
    Synchronize {
        src_layout: Option<vk::ImageLayout>,
        src_queue_family: Option<u32>,
        against: SmallVec<[QueueSubmission; 4]>,
    },
}

pub struct SynchronizationState<T: ResourceMarker> {
    owning_family: OptionalU32,
    layout: T::IfImage<vk::ImageLayout>,
    // FIXME for now we only allow exclusive access to resources, since when we are reading the state of global resources is already built
    // and it would be complicated to patch in more synchronization to allow Read Read overlap
    // it would be possible to do some compromise when the resource in only ever read in the whole graph
    access: SmallVec<[QueueSubmission; 4]>,
}

impl<T: ResourceMarker> SynchronizationState<T> {
    pub fn blank() -> Self {
        Self {
            owning_family: OptionalU32::NONE,
            layout: TypeOption::new_some(vk::ImageLayout::UNDEFINED),
            access: smallvec![],
        }
    }
    pub fn with_initial_layout(layout: vk::ImageLayout) -> Self {
        Self {
            layout: layout.into(),
            ..Self::blank()
        }
    }
    pub fn update_state(
        &mut self,
        // the initial state of the resource
        dst_family: u32,
        dst_layout: T::IfImage<vk::ImageLayout>,
        // the state of the resource at the end of the scheduled work
        final_access: &[QueueSubmission],
        final_layout: T::IfImage<vk::ImageLayout>,
        final_family: u32,
        // whether the resource was created with VK_ACCESS_MODE_CONCURRENT and does not need queue ownership transitions
        resource_concurrent: bool,
    ) -> SynchronizeResult
    where
        T::IfImage<vk::ImageLayout>: Eq + Copy,
    {
        assert!(!final_access.is_empty());

        let mut transition_layout = false;
        let mut transition_ownership = false;

        if T::IS_IMAGE && self.layout != dst_layout {
            transition_layout = true;
        }

        if !resource_concurrent
            && self.owning_family.is_some()
            && self.owning_family.unwrap() != dst_family
        {
            transition_ownership = true;
        }

        let result = if self.access.is_empty() {
            SynchronizeResult::None
        } else {
            SynchronizeResult::Synchronize {
                src_layout: transition_layout.then(|| self.layout.unwrap()),
                src_queue_family: transition_ownership.then(|| self.owning_family.unwrap()),
                against: self.access.clone(),
            }
        };

        self.owning_family = OptionalU32::new_some(dst_family);
        self.layout = dst_layout;
        self.access.clear();
        self.access.extend(final_access.iter().cloned());

        result
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

pub(crate) struct ImageViewEntry<T: ResourceMarker> {
    pub(crate) handle: T::EitherOut<vk::ImageView, vk::BufferView>,
    pub(crate) info_hash: u32,
    pub(crate) last_use: GenerationId,
}

pub struct ResourceMutableState<T: ResourceMarker> {
    pub(crate) views: SmallVec<[ImageViewEntry<T>; 2]>,
    pub(crate) synchronization: SynchronizationState<T>,
}

impl ResourceMutableState<ImageMarker> {
    pub fn with_initial_layout(layout: vk::ImageLayout) -> Self {
        Self {
            views: SmallVec::new(),
            synchronization: SynchronizationState::with_initial_layout(layout),
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

#[derive(Clone)]
pub struct Image(pub(crate) ArcHandle<Self>);
impl Object for Image {
    type CreateInfo = ImageCreateInfo;
    type SupplementalInfo = pumice_vma::AllocationCreateInfo;
    type Handle = vk::Image;
    type Storage = SimpleStorage<Self>;
    type ObjectData = (
        pumice_vma::Allocation,
        MutableShared<ResourceMutableState<ImageMarker>>,
    );

    type Parent = Device;

    unsafe fn create(
        ctx: &Device,
        info: &Self::CreateInfo,
        allocation_info: &Self::SupplementalInfo,
    ) -> VulkanResult<(Self::Handle, Self::ObjectData)> {
        let image_info = info.to_vk();
        ctx.allocator
            .create_image(&image_info, allocation_info)
            .map(|(handle, allocation, _)| {
                (
                    handle,
                    (
                        allocation,
                        MutableShared::new(ResourceMutableState::with_initial_layout(
                            info.initial_layout,
                        )),
                    ),
                )
            })
    }

    unsafe fn destroy(
        ctx: &Self::Parent,
        handle: Self::Handle,
        header: &ObjectHeader<Self>,
        _: &SynchronizationLock,
    ) -> VulkanResult<()> {
        ctx.allocator.destroy_image(handle, header.object_data.0);
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
        let mut data = header.object_data.1.get_mut(header.get_lock());
        data.get_view(header.handle, info, batch_id, self.0.get_parent())
    }
}
