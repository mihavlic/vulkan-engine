use std::ptr;

use super::{ArcHandle, Object};
use crate::context::device::Device;
use crate::{storage::nostore::SimpleStorage, submission::ReaderWriterState};
use pumice::util::impl_macros::ObjectHandle;
use pumice::util::result::VulkanResult;
use pumice::vk;
use smallvec::SmallVec;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SwapchainCreateInfo {
    pub surface: vk::SurfaceKHR,
    pub flags: vk::SwapchainCreateFlagsKHR,
    pub min_image_count: u32,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub extent: vk::Extent2D,
    pub array_layers: u32,
    pub usage: vk::ImageUsageFlags,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub present_mode: vk::PresentModeKHR,
    pub clipped: bool,
}

impl SwapchainCreateInfo {
    pub fn as_raw(&self) -> vk::SwapchainCreateInfoKHR {
        vk::SwapchainCreateInfoKHR {
            p_next: ptr::null(),
            flags: self.flags,
            surface: self.surface,
            min_image_count: self.min_image_count,
            image_format: self.format,
            image_color_space: self.color_space,
            image_extent: self.extent.clone(),
            image_array_layers: self.array_layers,
            image_usage: self.usage,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            pre_transform: self.pre_transform,
            composite_alpha: self.composite_alpha,
            present_mode: self.present_mode,
            clipped: self.clipped as u32,
            old_swapchain: vk::SwapchainKHR::null(),
            ..Default::default()
        }
    }
}

pub struct Swapchain(pub(crate) ArcHandle<Self>);
impl Object for Swapchain {
    type CreateInfo = SwapchainCreateInfo;
    type SupplementalInfo = ();
    type Handle = vk::SwapchainKHR;
    type Storage = SimpleStorage<Self>;
    type ObjectData = ();

    type Parent = Device;

    unsafe fn create(
        ctx: &Device,
        info: &Self::CreateInfo,
        allocation_info: &Self::SupplementalInfo,
    ) -> VulkanResult<(Self::Handle, Self::ObjectData)> {
        let info = info.as_raw();
        ctx.device
            .create_swapchain_khr(&info, ctx.instance.allocator_callbacks())
            .map(|handle| (handle, ()))
    }

    unsafe fn destroy(
        ctx: &Device,
        handle: Self::Handle,
        _: &Self::ObjectData,
    ) -> VulkanResult<()> {
        ctx.device
            .destroy_swapchain_khr(handle, ctx.instance.allocator_callbacks());
        VulkanResult::new_ok(())
    }

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        todo!()
    }
}
