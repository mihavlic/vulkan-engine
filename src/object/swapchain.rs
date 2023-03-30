use std::cell::RefMut;
use std::ptr;

use super::{ImageMutableState, ImageViewCreateInfo, ObjHandle, Object, ObjectData};

use crate::device::batch::GenerationId;
use crate::device::Device;
use crate::storage::nostore::SimpleStorage;
use crate::storage::{MutableShared, SynchronizationLock};
use pumice::util::ObjectHandle;
use pumice::vk;
use pumice::vk10::Extent2D;
use pumice::VulkanResult;

#[derive(PartialEq, Eq, Hash)]
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
    Stashed(StashedImage),
    Err(vk::Result),
}

pub(crate) struct SwapchainImage {
    pub image: vk::Image,
    pub state: ImageMutableState,
}

impl SwapchainImage {
    unsafe fn new(image: vk::Image, _ctx: &Device) -> Self {
        let _fence_info = vk::FenceCreateInfo {
            flags: pumice::vk10::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        let _semaphore_info = vk::SemaphoreCreateInfo::default();
        Self {
            image,
            state: ImageMutableState::new(vk::ImageLayout::UNDEFINED),
        }
    }
    unsafe fn destroy(&mut self, ctx: &Device) {
        self.state.destroy(ctx);
    }
}

#[derive(Clone)]
pub struct StashedImage {
    pub index: u32,
    pub semaphore: vk::Semaphore,
    pub fence: vk::Fence,
    pub suboptimal: bool,
}

pub(crate) struct SwapchainMutableState {
    swapchain: vk::SwapchainKHR,
    current_extent: vk::Extent2D,
    // signals that the surace has been resized and we need to recreate it the next time we acquire an image
    // this extent overrides the extent provided in SwapchainCreateInfo
    next_out_of_date: bool,
    resized_to: Option<vk::Extent2D>,
    images: Vec<SwapchainImage>,
    stashed_image: Option<StashedImage>,
}

impl SwapchainMutableState {
    pub unsafe fn new(create_info: &SwapchainCreateInfo, ctx: &Device) -> VulkanResult<Self> {
        let mut state = Self {
            swapchain: vk::SwapchainKHR::null(),
            current_extent: vk::Extent2D::default(),
            resized_to: None,
            images: Vec::new(),
            next_out_of_date: false,
            stashed_image: None,
        };

        state.recreate(create_info, ctx).map(|_| state)
    }
    /// Before calling this function, you must ensure that no access to the swapchain is occuring, whatever that means
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

        // spec states:
        //   "currentExtent [...] special value (0xFFFFFFFF, 0xFFFFFFFF) indicating that the surface size will be determined by the extent of a swapchain targeting the surface"
        let extent = if surface_info.current_extent.width == u32::MAX {
            let target_extent = self
                .resized_to
                .clone()
                .unwrap_or(create_info.extent.clone());

            let width = target_extent.width.clamp(
                surface_info.min_image_extent.width,
                surface_info.max_image_extent.width,
            );
            let height = target_extent.height.clamp(
                surface_info.min_image_extent.height,
                surface_info.max_image_extent.height,
            );

            vk::Extent2D { width, height }
        } else {
            surface_info.current_extent.clone()
        };

        self.current_extent = extent.clone();
        self.resized_to = None;
        self.next_out_of_date = false;
        assert!(
            self.stashed_image.is_none(),
            "Destroying a swapchain while its images are acquired is disallowed"
        );

        let max_image_count = if surface_info.max_image_count == 0 {
            u32::MAX
        } else {
            surface_info.max_image_count
        };

        let create_info = vk::SwapchainCreateInfoKHR {
            image_extent: extent,
            old_swapchain: self.swapchain,
            min_image_count: create_info
                .min_image_count
                .clamp(surface_info.min_image_count, max_image_count),
            ..create_info.to_vk()
        };

        for data in &mut self.images {
            std::mem::replace(
                &mut data.state,
                ImageMutableState::new(vk::ImageLayout::UNDEFINED),
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
            for mut image in self.images.drain(new_len..) {
                image.destroy(ctx);
            }
        }

        for (old_image, new_image) in self.images.iter_mut().zip(images) {
            std::mem::replace(
                &mut old_image.state,
                ImageMutableState::new(vk::ImageLayout::UNDEFINED),
            )
            .destroy(ctx);
            old_image.image = new_image;
        }

        Ok(())
    }
    pub unsafe fn acquire_image(
        &mut self,
        timeout: u64,
        arguments: impl FnOnce() -> (vk::Semaphore, vk::Fence),
        stash_acquired: bool,
        ctx: &Device,
    ) -> SwapchainAcquireStatus {
        if let Some(stashed) = self.stashed_image.clone() {
            if !stash_acquired {
                self.stashed_image = None;
            }

            return SwapchainAcquireStatus::Stashed(stashed);
        }

        if self.resized_to.is_some() || self.next_out_of_date {
            return SwapchainAcquireStatus::OutOfDate;
        }

        let (semaphore, fence) = arguments();

        let result = ctx
            .device()
            .acquire_next_image_khr(self.swapchain, timeout, semaphore, fence);

        match result {
            Ok((index, res @ (vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR))) => {
                let suboptimal = res == vk::Result::SUBOPTIMAL_KHR;
                if stash_acquired {
                    self.stash_acquired_image(StashedImage {
                        index,
                        semaphore,
                        fence,
                        suboptimal,
                    })
                }
                SwapchainAcquireStatus::Ok(index, suboptimal)
            }
            Ok((_, vk::Result::TIMEOUT | vk::Result::NOT_READY)) => {
                return SwapchainAcquireStatus::Timeout
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return SwapchainAcquireStatus::OutOfDate,
            Err(err) => return SwapchainAcquireStatus::Err(err),
            Ok((_, res)) => unreachable!("Invalid Ok result value: {:?}", res),
        }
    }
    pub fn stash_acquired_image(&mut self, stash: StashedImage) {
        assert!(self.stashed_image.is_none());
        self.stashed_image = Some(stash);
    }
    pub unsafe fn clear_stashed_image(&mut self) {
        self.stashed_image = None;
    }
    pub unsafe fn get_view(
        &mut self,
        image_index: u32,
        info: &ImageViewCreateInfo,
        batch_id: GenerationId,
        device: &Device,
    ) -> VulkanResult<vk::ImageView> {
        let swapchain_image = &mut self.images[image_index as usize];
        swapchain_image
            .state
            .get_view(swapchain_image.image, info, batch_id, None, device)
    }
    pub fn surface_resized(&mut self, new_extent: vk::Extent2D) {
        self.resized_to = Some(new_extent);
    }
    pub fn set_next_out_of_date(&mut self) {
        self.next_out_of_date = true;
    }
    pub fn get_extent(&self) -> Extent2D {
        self.resized_to
            .clone()
            .unwrap_or(self.current_extent.clone())
    }
    pub fn raw(&self) -> vk::SwapchainKHR {
        self.swapchain
    }
    pub fn get_image_data(&self, image_index: u32) -> &SwapchainImage {
        &self.images[image_index as usize]
    }
    pub unsafe fn destroy(&mut self, ctx: &Device) {
        if self.swapchain != vk::SwapchainKHR::null() {
            ctx.device()
                .destroy_swapchain_khr(self.swapchain, ctx.allocator_callbacks());
        }
        for mut data in self.images.drain(..) {
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

pub struct SwapchainState {
    info: SwapchainCreateInfo,
    mutable: MutableShared<SwapchainMutableState>,
}

impl SwapchainState {
    pub(crate) unsafe fn get_mutable_state<'a>(
        &'a self,
        lock: &'a SynchronizationLock,
    ) -> RefMut<'a, SwapchainMutableState> {
        self.mutable.get_mut(lock)
    }
}

impl ObjectData for SwapchainState {
    type CreateInfo = SwapchainCreateInfo;
    type Handle = ();

    fn get_create_info(&self) -> &Self::CreateInfo {
        &self.info
    }
    fn get_handle(&self) -> Self::Handle {
        ()
    }
}

create_object! {Swapchain}
impl Object for Swapchain {
    type Storage = SimpleStorage<Self>;
    type Parent = Device;

    type InputData<'a> = SwapchainCreateInfo;
    type Data = SwapchainState;

    unsafe fn create<'a>(
        data: Self::InputData<'a>,
        ctx: &Self::Parent,
    ) -> VulkanResult<Self::Data> {
        SwapchainMutableState::new(&data, ctx).map(|state| SwapchainState {
            info: data,
            mutable: MutableShared::new(state),
        })
    }

    unsafe fn destroy(
        data: &Self::Data,
        lock: &SynchronizationLock,
        ctx: &Self::Parent,
    ) -> VulkanResult<()> {
        data.mutable.get_mut(lock).destroy(ctx);
        ctx.instance
            .handle()
            .destroy_surface_khr(data.info.surface, ctx.allocator_callbacks());

        VulkanResult::Ok(())
    }

    fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.swapchain_storage
    }
}

impl Swapchain {
    pub unsafe fn surface_resized(&self, new_extent: vk::Extent2D) {
        self.0
            .access_mutable(|d| &d.mutable, |s| s.surface_resized(new_extent))
    }
    pub unsafe fn get_view(
        &self,
        image_index: u32,
        info: &ImageViewCreateInfo,
        batch_id: GenerationId,
    ) -> VulkanResult<vk::ImageView> {
        let parent = self.0.get_parent();
        self.0.access_mutable(
            |d| &d.mutable,
            |s| s.get_view(image_index, info, batch_id, parent),
        )
    }
    pub fn get_extent(&self) -> vk::Extent2D {
        unsafe { self.0.access_mutable(|d| &d.mutable, |s| s.get_extent()) }
    }
}
