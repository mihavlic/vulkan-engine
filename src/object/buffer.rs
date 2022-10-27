use std::ptr;

use crate::{
    arena::optional::OptionalU32, context::device::InnerDevice, storage::nostore::NoStore,
    synchronization::ReaderWriterState,
};
use pumice::{util::result::VulkanResult, vk};
use smallvec::SmallVec;

use super::{ArcHandle, Object};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferCreateInfo {
    flags: vk::BufferCreateFlags,
    size: u64,
    usage: vk::BufferUsageFlags,
}

impl BufferCreateInfo {
    pub fn as_raw(&self) -> vk::BufferCreateInfo {
        vk::BufferCreateInfo {
            p_next: ptr::null(),
            flags: self.flags,
            size: self.size,
            usage: self.usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferSynchronizationState {
    owning_family: OptionalU32,
    state: ReaderWriterState,
}

impl BufferSynchronizationState {
    pub const BLANK: Self = Self {
        owning_family: OptionalU32::NONE,
        state: ReaderWriterState::None,
    };
}

pub struct Buffer(pub(crate) ArcHandle<Self>);
impl Object for Buffer {
    type CreateInfo = BufferCreateInfo;
    type SupplementalInfo = pumice_vma::AllocationCreateInfo;
    type Handle = vk::Buffer;
    type Storage = NoStore;
    type ObjectData = (pumice_vma::Allocation, BufferSynchronizationState);

    type Parent = InnerDevice;

    unsafe fn create(
        ctx: &InnerDevice,
        info: &Self::CreateInfo,
        allocation_info: &Self::SupplementalInfo,
    ) -> VulkanResult<(Self::Handle, Self::ObjectData)> {
        let image_info = info.as_raw();
        ctx.allocator
            .create_buffer(&image_info, allocation_info)
            .map(|(handle, allocation, _)| {
                (handle, (allocation, BufferSynchronizationState::BLANK))
            })
    }

    unsafe fn destroy(
        ctx: &InnerDevice,
        handle: Self::Handle,
        &(allocation, _): &Self::ObjectData,
    ) -> VulkanResult<()> {
        ctx.allocator.destroy_buffer(handle, allocation);
        VulkanResult::new_ok(())
    }

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.buffer_storage
    }
}
