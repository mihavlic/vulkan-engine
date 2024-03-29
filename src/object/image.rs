use std::{
    borrow::Cow,
    fmt::Display,
    hash::{Hash, Hasher},
    ptr,
};

use pumice::{vk, VulkanResult};
use smallvec::{smallvec, SmallVec};

use crate::{
    arena::uint::OptionalU32,
    device::{
        batch::GenerationId,
        debug::{maybe_attach_debug_label, LazyDisplay},
        submission::QueueSubmission,
        Device,
    },
    graph::resource_marker::{ImageMarker, ResourceMarker, TypeOption, TypeSome},
    storage::{constant_ahash_hasher, nostore::SimpleStorage, MutableShared, SynchronizationLock},
};

use super::{ObjHandle, ObjRef, Object, ObjectData};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Extent {
    D1(u32),
    D2(u32, u32),
    D3(u32, u32, u32),
}

impl Default for Extent {
    fn default() -> Self {
        Self::D2(0, 0)
    }
}

impl Extent {
    pub fn as_image_type(&self) -> vk::ImageType {
        match self {
            Extent::D1(_) => vk::ImageType::T1D,
            Extent::D2(_, _) => vk::ImageType::T2D,
            Extent::D3(_, _, _) => vk::ImageType::T3D,
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
    pub fn get_1d(&self) -> Option<u32> {
        if let Self::D1(x) = *self {
            return Some(x);
        }
        None
    }
    pub fn get_2d(&self) -> Option<(u32, u32)> {
        if let Self::D2(x, y) = *self {
            return Some((x, y));
        }
        None
    }
    pub fn get_3d(&self) -> Option<(u32, u32, u32)> {
        if let Self::D3(x, y, z) = *self {
            return Some((x, y, z));
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
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
    pub label: Option<Cow<'static, str>>,
}

impl ImageCreateInfo {
    pub(crate) fn to_vk(&self) -> vk::ImageCreateInfo {
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

pub struct SynchronizeResult {
    pub transition_layout_from: Option<vk::ImageLayout>,
    pub transition_ownership_from: Option<u32>,
    pub prev_access: SmallVec<[QueueSubmission; 4]>,
}

pub enum HostAccessKind {
    Immediate,
    Synchronized(QueueSubmission),
}

pub struct InnerSynchronizationState<T: ResourceMarker> {
    pub owning_family: OptionalU32,
    pub layout: T::IfImage<vk::ImageLayout>,
    // FIXME for now we only allow exclusive access to resources, since when we are reading the state of global resources is already built
    // and it would be complicated to patch in more synchronization to allow Read Read overlap
    // it would be possible to do some compromise when the resource in only ever read in the whole graph
    pub access: SmallVec<[QueueSubmission; 4]>,
}

impl<T: ResourceMarker> InnerSynchronizationState<T> {
    pub(crate) fn blank() -> Self {
        Self {
            owning_family: OptionalU32::NONE,
            layout: TypeOption::new_some(vk::ImageLayout::UNDEFINED),
            access: smallvec![],
        }
    }
    pub(crate) fn with_initial_layout(layout: vk::ImageLayout) -> Self {
        Self {
            layout: layout.into(),
            ..Self::blank()
        }
    }
    pub(crate) fn retain_active_submissions(&mut self, device: &Device) {
        device.retain_active_submissions(&mut self.access)
    }
    pub(crate) fn update(
        &mut self,
        // the initial state of the resource
        dst_family: u32,
        dst_layout: T::IfImage<vk::ImageLayout>,
        // the state of the resource at the end of the scheduled work
        final_accessors: &[QueueSubmission],
        final_layout: T::IfImage<vk::ImageLayout>,
        final_family: u32,
        // whether the resource was created with VK_ACCESS_MODE_CONCURRENT and does not need queue ownership transitions
        resource_concurrent: bool,
    ) -> SynchronizeResult
    where
        T::IfImage<vk::ImageLayout>: Eq + Copy,
    {
        assert!(!final_accessors.is_empty());

        let mut transition_layout_from = None;
        let mut transition_ownership_from = None;

        if T::IS_IMAGE
            && dst_layout.unwrap() != vk::ImageLayout::UNDEFINED
            && self.layout != dst_layout
        {
            transition_layout_from = Some(self.layout.unwrap());
        }

        if !resource_concurrent
            && self.owning_family.is_some()
            && (T::IS_BUFFER || dst_layout.unwrap() != vk::ImageLayout::UNDEFINED)
            && self.owning_family.unwrap() != dst_family
        {
            transition_ownership_from = Some(self.owning_family.unwrap());
        }

        let result = SynchronizeResult {
            transition_layout_from,
            transition_ownership_from,
            prev_access: self.access.clone(),
        };

        self.owning_family = OptionalU32::new_some(final_family);
        self.layout = final_layout;
        self.access.clear();
        self.access.extend(final_accessors.iter().cloned());

        result
    }
    pub(crate) fn update_host_access(
        &mut self,
        update_fn: impl FnOnce(&[QueueSubmission]) -> HostAccessKind,
    ) -> SmallVec<[QueueSubmission; 4]> {
        let prev_accessors = self.access.clone();
        self.access.clear();

        let kind = update_fn(&prev_accessors);

        match kind {
            HostAccessKind::Immediate => {}
            HostAccessKind::Synchronized(submission) => {
                self.access.push(submission);
            }
        }

        prev_accessors
    }
}

pub struct SynchronizationState<T: ResourceMarker>(InnerSynchronizationState<T>);

impl<T: ResourceMarker> SynchronizationState<T> {
    pub(crate) fn blank() -> Self {
        Self(InnerSynchronizationState::blank())
    }
    pub(crate) fn with_initial_layout(layout: vk::ImageLayout) -> Self {
        Self(InnerSynchronizationState::with_initial_layout(layout))
    }
    pub fn retain_active_submissions(&mut self, device: &Device) {
        self.0.retain_active_submissions(device)
    }
    pub fn update(
        &mut self,
        // the initial state of the resource
        dst_family: u32,
        dst_layout: T::IfImage<vk::ImageLayout>,
        // the state of the resource at the end of the scheduled work
        final_accessors: &[QueueSubmission],
        final_layout: T::IfImage<vk::ImageLayout>,
        final_family: u32,
        // whether the resource was created with VK_ACCESS_MODE_CONCURRENT and does not need queue ownership transitions
        resource_concurrent: bool,
    ) -> SynchronizeResult
    where
        T::IfImage<vk::ImageLayout>: Eq + Copy,
    {
        self.0.update(
            dst_family,
            dst_layout,
            final_accessors,
            final_layout,
            final_family,
            resource_concurrent,
        )
    }
    /// performs host access which is finished while the resource lock is held
    /// the accessor must wait for the submissions returned
    pub fn update_host_access(
        &mut self,
        update_fn: impl FnOnce(&[QueueSubmission]) -> HostAccessKind,
    ) -> SmallVec<[QueueSubmission; 4]> {
        self.0.update_host_access(update_fn)
    }
    /// allows the caller to modify the inner state however they want
    /// it is up to the caller to leave it in a valid state
    pub unsafe fn get_unchecked_mut(&mut self) -> &mut InnerSynchronizationState<T> {
        &mut self.0
    }
}

#[derive(Clone, Hash)]
pub struct ImageViewCreateInfo {
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub components: vk::ComponentMapping,
    pub subresource_range: vk::ImageSubresourceRange,
}

impl ImageViewCreateInfo {
    pub fn to_vk(&self, image: vk::Image) -> vk::ImageViewCreateInfo {
        vk::ImageViewCreateInfo {
            image,
            view_type: self.view_type,
            format: self.format,
            components: self.components.clone(),
            subresource_range: self.subresource_range.clone(),
            ..Default::default()
        }
    }
    pub fn get_hash(&self) -> u32 {
        let mut hasher = constant_ahash_hasher();
        self.hash(&mut hasher);

        // this is dumb but with a high quality function truncating like this should be kind of ok
        hasher.finish() as u32
    }
}

pub(crate) struct ImageViewEntry {
    handle: vk::ImageView,
    info_hash: u32,
    last_use: GenerationId,
}

pub struct ImageMutableState {
    views: SmallVec<[ImageViewEntry; 2]>,
    synchronization: SynchronizationState<ImageMarker>,
}

impl ImageMutableState {
    pub(crate) fn new(layout: vk::ImageLayout) -> Self {
        Self {
            views: SmallVec::new(),
            synchronization: SynchronizationState::with_initial_layout(layout),
        }
    }
    pub fn synchronization_state(&mut self) -> &mut SynchronizationState<ImageMarker> {
        &mut self.synchronization
    }
    pub unsafe fn get_view(
        &mut self,
        self_handle: vk::Image,
        info: &ImageViewCreateInfo,
        batch_id: GenerationId,
        label: Option<&dyn Display>,
        device: &Device,
    ) -> VulkanResult<vk::ImageView> {
        let hash = info.get_hash();

        if let Some(found) = self.views.iter_mut().find(|v| v.info_hash == hash) {
            found.last_use = batch_id;
            VulkanResult::Ok(found.handle)
        } else {
            let raw = info.to_vk(self_handle);

            let view = device
                .device()
                .create_image_view(&raw, device.allocator_callbacks())?;

            if let Some(label) = label {
                maybe_attach_debug_label(view, label, device);
            }

            let entry = ImageViewEntry {
                handle: view,
                info_hash: hash,
                last_use: batch_id,
            };

            self.views.push(entry);

            VulkanResult::Ok(view)
        }
    }
    pub(crate) unsafe fn destroy(&mut self, ctx: &Device) {
        for view in self.views.drain(..) {
            ctx.device()
                .destroy_image_view(view.handle, ctx.allocator_callbacks());
        }
    }
}

pub struct ImageState {
    handle: vk::Image,
    info: ImageCreateInfo,
    allocation: pumice_vma::Allocation,
    mutable: MutableShared<ImageMutableState>,
}

impl ImageState {
    pub unsafe fn get_mutable_state(&self) -> &MutableShared<ImageMutableState> {
        &self.mutable
    }
    pub unsafe fn update_state(
        &self,
        // the initial state of the resource
        dst_family: u32,
        dst_layout: vk::ImageLayout,
        // the state of the resource at the end of the scheduled work
        final_access: &[QueueSubmission],
        final_layout: vk::ImageLayout,
        final_family: u32,
        // whether the resource was created with VK_ACCESS_MODE_CONCURRENT and does not need queue ownership transitions
        resource_concurrent: bool,
        lock: &SynchronizationLock,
    ) -> SynchronizeResult {
        self.mutable.get_mut(lock).synchronization.update(
            dst_family,
            TypeSome::new_some(dst_layout),
            final_access,
            TypeSome::new_some(final_layout),
            final_family,
            resource_concurrent,
        )
    }
}

impl ObjectData for ImageState {
    type CreateInfo = ImageCreateInfo;
    type Handle = vk::Image;

    fn get_create_info(&self) -> &Self::CreateInfo {
        &self.info
    }
    fn get_handle(&self) -> Self::Handle {
        self.handle
    }
}

create_object! {Image}
derive_raw_handle! {Image, vk::Image}
impl Object for Image {
    type Storage = SimpleStorage<Self>;
    type Parent = Device;

    type InputData<'a> = (ImageCreateInfo, pumice_vma::AllocationCreateInfo);
    type Data = ImageState;

    unsafe fn create<'a>(
        data: Self::InputData<'a>,
        ctx: &Self::Parent,
    ) -> VulkanResult<Self::Data> {
        let image_info = data.0.to_vk();
        ctx.allocator
            .create_image(&image_info, &data.1)
            .map(|(handle, allocation, _)| {
                if let Some(label) = data.0.label.as_ref() {
                    maybe_attach_debug_label(handle, &label, ctx);
                }
                ImageState {
                    handle,
                    mutable: MutableShared::new(ImageMutableState::new(data.0.initial_layout)),
                    info: data.0,
                    allocation,
                }
            })
    }
    unsafe fn destroy(
        data: &Self::Data,
        lock: &SynchronizationLock,
        ctx: &Self::Parent,
    ) -> VulkanResult<()> {
        data.mutable.get_mut(lock).destroy(ctx);
        ctx.allocator.destroy_image(data.handle, data.allocation);
        VulkanResult::Ok(())
    }

    fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.image_storage
    }
}

impl ObjRef<Image> {
    pub fn get_allocation(&self) -> &pumice_vma::Allocation {
        &self.get_object_data().allocation
    }
    pub unsafe fn get_view(
        &self,
        info: &ImageViewCreateInfo,
        batch_id: GenerationId,
    ) -> VulkanResult<vk::ImageView> {
        let device = self.get_parent();
        let data = self.get_object_data();

        let label = LazyDisplay(|f| {
            write!(
                f,
                "{} view {:x}",
                data.get_create_info().label.as_deref().unwrap_or("Image"),
                info.get_hash()
            )
        });

        self.access_mutable(
            |d| &d.mutable,
            |m| m.get_view(data.handle, info, batch_id, Some(&label), device),
        )
    }
    pub fn get_whole_subresource_range(&self) -> vk::ImageSubresourceRange {
        let info = self.get_create_info();
        vk::ImageSubresourceRange {
            aspect_mask: info.format.get_format_aspects().0,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }
}
