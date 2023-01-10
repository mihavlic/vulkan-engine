use std::{
    hash::{Hash, Hasher},
    ptr,
};

use crate::{
    arena::uint::OptionalU32,
    batch::GenerationId,
    context::device::Device,
    storage::{constant_ahash_hasher, nostore::SimpleStorage, MutableShared},
    submission::ReaderWriterState,
};
use pumice::{vk, VulkanResult};
use smallvec::SmallVec;

use super::{ArcHandle, Object};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferCreateInfo {
    flags: vk::BufferCreateFlags,
    size: u64,
    usage: vk::BufferUsageFlags,
    pub sharing_mode_concurrent: bool,
}

impl BufferCreateInfo {
    pub fn to_vk(&self) -> vk::BufferCreateInfo {
        vk::BufferCreateInfo {
            p_next: ptr::null(),
            flags: self.flags,
            size: self.size,
            usage: self.usage,
            sharing_mode: vk::SharingMode(self.sharing_mode_concurrent as _),
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

#[derive(Clone, Hash)]
pub struct BufferViewCreateInfo {
    pub flags: vk::BufferViewCreateFlags,
    pub buffer: vk::Buffer,
    pub format: vk::Format,
    pub offset: vk::DeviceSize,
    pub range: vk::DeviceSize,
}

impl BufferViewCreateInfo {
    fn get_hash(&self) -> u32 {
        let mut hasher = constant_ahash_hasher();
        self.hash(&mut hasher);

        // this is dumb but with a high quality function truncating like this should be kind of ok
        hasher.finish() as u32
    }
}

struct BufferViewEntry {
    handle: vk::BufferView,
    info_hash: u32,
    last_use: GenerationId,
}

pub struct BufferMutableState {
    views: SmallVec<[BufferViewEntry; 2]>,
    synchronization: BufferSynchronizationState,
}

impl BufferMutableState {
    pub fn new() -> Self {
        Self {
            views: SmallVec::new(),
            synchronization: BufferSynchronizationState::BLANK,
        }
    }
    pub unsafe fn get_view(
        &mut self,
        self_handle: vk::Buffer,
        info: &BufferViewCreateInfo,
        batch_id: GenerationId,
        device: &Device,
    ) -> VulkanResult<vk::BufferView> {
        let hash = info.get_hash();

        if let Some(found) = self.views.iter_mut().find(|v| v.info_hash == hash) {
            found.last_use = batch_id;
            VulkanResult::Ok(found.handle)
        } else {
            let raw = vk::BufferViewCreateInfo {
                flags: info.flags,
                buffer: info.buffer,
                format: info.format,
                offset: info.offset,
                range: info.range,
                ..Default::default()
            };

            let view = device
                .device()
                .create_buffer_view(&raw, device.allocator_callbacks())?;

            let entry = BufferViewEntry {
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
                .destroy_buffer_view(view.handle, device.allocator_callbacks());
        }
    }
}

#[derive(Clone)]
pub struct Buffer(pub(crate) ArcHandle<Self>);
impl Object for Buffer {
    type CreateInfo = BufferCreateInfo;
    type SupplementalInfo = pumice_vma::AllocationCreateInfo;
    type Handle = vk::Buffer;
    type Storage = SimpleStorage<Self>;
    type ObjectData = (pumice_vma::Allocation, MutableShared<BufferMutableState>);

    type Parent = Device;

    unsafe fn create(
        ctx: &Device,
        info: &Self::CreateInfo,
        allocation_info: &Self::SupplementalInfo,
    ) -> VulkanResult<(Self::Handle, Self::ObjectData)> {
        let image_info = info.to_vk();
        ctx.allocator
            .create_buffer(&image_info, allocation_info)
            .map(|(handle, allocation, _)| {
                (
                    handle,
                    (allocation, MutableShared::new(BufferMutableState::new())),
                )
            })
    }

    unsafe fn destroy(
        ctx: &Device,
        handle: Self::Handle,
        &(allocation, _): &Self::ObjectData,
    ) -> VulkanResult<()> {
        ctx.allocator.destroy_buffer(handle, allocation);
        VulkanResult::Ok(())
    }

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.buffer_storage
    }
}
