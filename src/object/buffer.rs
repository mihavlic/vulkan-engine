use std::{
    hash::{Hash, Hasher},
    ptr,
};

use crate::{
    arena::uint::OptionalU32,
    device::{
        batch::GenerationId,
        submission::{QueueSubmission, ReaderWriterState},
        Device,
    },
    graph::resource_marker::{BufferMarker, TypeNone, TypeOption},
    storage::{constant_ahash_hasher, nostore::SimpleStorage, MutableShared, SynchronizationLock},
};
use pumice::{vk, VulkanResult};
use smallvec::SmallVec;

use super::{ObjHandle, Object, ObjectData, SynchronizationState, SynchronizeResult};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferCreateInfo {
    pub flags: vk::BufferCreateFlags,
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
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

pub(crate) struct BufferViewEntry {
    handle: vk::BufferView,
    info_hash: u32,
    last_use: GenerationId,
}

pub struct BufferMutableState {
    views: SmallVec<[BufferViewEntry; 2]>,
    synchronization: SynchronizationState<BufferMarker>,
}

impl BufferMutableState {
    pub fn new() -> Self {
        Self {
            views: SmallVec::new(),
            synchronization: SynchronizationState::blank(),
        }
    }
    pub(crate) fn get_synchronization_state(&mut self) -> &mut SynchronizationState<BufferMarker> {
        &mut self.synchronization
    }
    pub unsafe fn get_view(
        &mut self,
        _self_handle: vk::Buffer,
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
    pub unsafe fn destroy(&mut self, ctx: &Device) {
        for view in self.views.drain(..) {
            ctx.device()
                .destroy_buffer_view(view.handle, ctx.allocator_callbacks());
        }
    }
}

pub struct BufferState {
    handle: vk::Buffer,
    info: BufferCreateInfo,
    allocation: pumice_vma::Allocation,
    mutable: MutableShared<BufferMutableState>,
}

impl BufferState {
    pub unsafe fn get_mutable_state(&self) -> &MutableShared<BufferMutableState> {
        &self.mutable
    }
    pub unsafe fn update_state(
        &self,
        // the initial state of the resource
        dst_family: u32,
        // the state of the resource at the end of the scheduled work
        final_access: &[QueueSubmission],
        final_family: u32,
        // whether the resource was created with VK_ACCESS_MODE_CONCURRENT and does not need queue ownership transitions
        resource_concurrent: bool,
        lock: &SynchronizationLock,
    ) -> SynchronizeResult {
        self.mutable.get_mut(lock).synchronization.update_state(
            dst_family,
            TypeNone::new_none(),
            final_access,
            TypeNone::new_none(),
            final_family,
            resource_concurrent,
        )
    }
}

impl ObjectData for BufferState {
    type CreateInfo = BufferCreateInfo;
    type Handle = vk::Buffer;

    fn get_create_info(&self) -> &Self::CreateInfo {
        &self.info
    }
    fn get_handle(&self) -> Self::Handle {
        self.handle
    }
}
create_object! {Buffer}
derive_raw_handle! {Buffer, vk::Buffer}
impl Object for Buffer {
    type Storage = SimpleStorage<Self>;
    type Parent = Device;

    type InputData<'a> = (BufferCreateInfo, pumice_vma::AllocationCreateInfo);
    type Data = BufferState;

    unsafe fn create<'a>(
        data: Self::InputData<'a>,
        ctx: &Self::Parent,
    ) -> VulkanResult<Self::Data> {
        let buffer_info = data.0.to_vk();
        ctx.allocator
            .create_buffer(&buffer_info, &data.1)
            .map(|(handle, allocation, _)| BufferState {
                handle,
                info: data.0,
                allocation,
                mutable: MutableShared::new(BufferMutableState::new()),
            })
    }
    unsafe fn destroy(
        data: &Self::Data,
        lock: &SynchronizationLock,
        ctx: &Self::Parent,
    ) -> VulkanResult<()> {
        data.mutable.get_mut(lock).destroy(ctx);
        ctx.allocator.destroy_buffer(data.handle, data.allocation);
        VulkanResult::Ok(())
    }

    fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.buffer_storage
    }
}
