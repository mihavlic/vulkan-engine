use std::ptr;

use super::{ArcHandle, Object, ResourceMutableState};
use crate::arena::uint::{Config, PackedUint};
use crate::context::device::Device;
use crate::graph::resource_marker::ImageMarker;
use crate::storage::{MutableShared, ObjectHeader, SynchronizationLock};
use crate::{storage::nostore::SimpleStorage, submission::ReaderWriterState};
use pumice::util::ObjectHandle;
use pumice::vk;
use pumice::VulkanResult;
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

pub enum SwapchainAcquireStatus {
    OutOfDate,
    Timeout,
    // The spec states:
    //   If an image is acquired successfully, vkAcquireNextImageKHR must either return VK_SUCCESS or VK_SUBOPTIMAL_KHR.
    //   The implementation may return VK_SUBOPTIMAL_KHR if the swapchain no longer matches the surface properties exactly,
    //   but can still be used for presentation.
    // so VK_TIMEOUT and VK_NOT_READY do not actually acquire a valid image even though they are not error codes
    /// True if the acquired image is suboptimal
    Ok(u32, bool),
    Err(vk::Result),
}

pub(crate) struct SwapchainImage {
    pub image: vk::Image,
    pub state: ResourceMutableState<ImageMarker>,
}

impl SwapchainImage {
    unsafe fn new(image: vk::Image, ctx: &Device) -> Self {
        let fence_info = vk::FenceCreateInfo {
            flags: pumice::vk10::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        Self {
            image,
            state: ResourceMutableState::with_initial_layout(vk::ImageLayout::UNDEFINED),
        }
    }
    unsafe fn destroy(self, ctx: &Device) {
        self.state.destroy(ctx);
    }
}

pub(crate) struct SwapchainState {
    swapchain: vk::SwapchainKHR,
    extent: vk::Extent2D,
    images: Vec<SwapchainImage>,
}

impl SwapchainState {
    pub unsafe fn new(create_info: &SwapchainCreateInfo, ctx: &Device) -> VulkanResult<Self> {
        let mut state = Self {
            swapchain: vk::SwapchainKHR::null(),
            extent: vk::Extent2D::default(),
            images: Vec::new(),
        };

        state.recreate(create_info, ctx).map(|_| state)
    }
    /// Before calling this function, you must ensure that no access to the swapchain is occuring
    pub unsafe fn recreate(
        &mut self,
        create_info: &SwapchainCreateInfo,
        ctx: &Device,
    ) -> VulkanResult<()> {
        let surface_info = ctx
            .instance
            .handle()
            .get_physical_device_surface_capabilities_khr(
                ctx.physical_device,
                create_info.surface,
            )?;

        self.extent = surface_info.current_extent.clone();

        let create_info = vk::SwapchainCreateInfoKHR {
            image_extent: surface_info.current_extent,
            old_swapchain: self.swapchain,
            ..create_info.to_vk()
        };

        for data in &mut self.images {
            std::mem::replace(
                &mut data.state,
                ResourceMutableState::with_initial_layout(vk::ImageLayout::UNDEFINED),
            )
            .destroy(ctx);
        }

        self.swapchain = ctx
            .device()
            .create_swapchain_khr(&create_info, ctx.allocator_callbacks())?;

        if create_info.old_swapchain != vk::SwapchainKHR::null() {
            ctx.device()
                .destroy_swapchain_khr(create_info.old_swapchain, ctx.allocator_callbacks());
        }

        let (images, _) = ctx
            .device()
            .get_swapchain_images_khr(self.swapchain, None)?;

        let old_len = self.images.len();
        let new_len = images.len();
        if new_len > old_len {
            self.images
                .resize_with(new_len, || SwapchainImage::new(vk::Image::null(), ctx));
        } else if new_len < old_len {
            for image in self.images.drain(new_len..) {
                image.destroy(ctx);
            }
        }

        for (old_image, new_image) in self.images.iter_mut().zip(images) {
            std::mem::replace(
                &mut old_image.state,
                ResourceMutableState::with_initial_layout(vk::ImageLayout::UNDEFINED),
            )
            .destroy(ctx);
            old_image.image = new_image;
        }

        Ok(())
    }
    pub unsafe fn acquire_image(
        &mut self,
        timeout: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
        ctx: &Device,
    ) -> SwapchainAcquireStatus {
        let result = ctx
            .device()
            .acquire_next_image_khr(self.swapchain, timeout, semaphore, fence);

        match result {
            Ok((index, vk::Result::SUCCESS)) => SwapchainAcquireStatus::Ok(index, false),
            Ok((index, vk::Result::TIMEOUT | vk::Result::NOT_READY)) => {
                return SwapchainAcquireStatus::Timeout
            }
            Ok((index, vk::Result::SUBOPTIMAL_KHR)) => {
                return SwapchainAcquireStatus::Ok(index, true)
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return SwapchainAcquireStatus::OutOfDate,
            Err(err) => return SwapchainAcquireStatus::Err(err),
            Ok((_, res)) => unreachable!("Invalid Ok result value: {:?}", res),
        }
    }
    pub fn get_swapchain(&self, image_index: u32) -> vk::SwapchainKHR {
        self.swapchain
    }
    pub fn get_image_data(&self, image_index: u32) -> &SwapchainImage {
        &self.images[image_index as usize]
    }
    pub fn get_image_data_mut(&mut self, image_index: u32) -> &mut SwapchainImage {
        &mut self.images[image_index as usize]
    }
    pub unsafe fn destroy(self, ctx: &Device) {
        for data in self.images {
            data.destroy(ctx);
        }
    }
}

impl SwapchainCreateInfo {
    pub fn to_vk(&self) -> vk::SwapchainCreateInfoKHR {
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

#[derive(Clone)]
pub struct Swapchain(pub(crate) ArcHandle<Self>);
impl Object for Swapchain {
    type CreateInfo = SwapchainCreateInfo;
    type SupplementalInfo = ();
    type Handle = ();
    type Storage = SimpleStorage<Self>;
    type ObjectData = MutableShared<SwapchainState>;

    type Parent = Device;

    unsafe fn create(
        ctx: &Device,
        info: &Self::CreateInfo,
        allocation_info: &Self::SupplementalInfo,
    ) -> VulkanResult<(Self::Handle, Self::ObjectData)> {
        SwapchainState::new(info, ctx).map(|state| ((), MutableShared::new(state)))
    }

    unsafe fn destroy(
        ctx: &Self::Parent,
        handle: Self::Handle,
        header: &ObjectHeader<Self>,
        lock: &SynchronizationLock,
    ) -> VulkanResult<()> {
        ctx.device.destroy_swapchain_khr(
            header.object_data.get(lock).swapchain,
            ctx.instance.allocator_callbacks(),
        );

        ctx.instance
            .handle()
            .destroy_surface_khr(header.info.surface, ctx.allocator_callbacks());

        VulkanResult::Ok(())
    }

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        todo!()
    }
}

// impl Swapchain {
//     unsafe fn acquire_image(
//         &self,
//         timeout: u64,
//         semaphore: vk::Semaphore,
//         fence: vk::Fence,
//         ctx: &Device,
//     ) -> SwapchainAcquireStatus {
//         let handle = self.0.get_header().handle;
//         self.0.access_mutable(
//             |d| d,
//             |m| m.acquire_image(handle, timeout, semaphore, fence, ctx),
//         )
//     }
// }
