use std::ptr;

use crate::device::Device;
use crate::storage::nostore::SimpleStorage;
use crate::storage::SynchronizationLock;
use crate::util::ffi_ptr::AsFFiPtr;

use super::{ArcHandle, Object};
use super::{BasicObjectData, ObjectData};
use pumice::util::ObjectHandle;
use pumice::vk;
use pumice::VulkanResult;
use smallvec::SmallVec;

pub struct PipelineLayoutCreateInfo {
    pub flags: vk::PipelineLayoutCreateFlags,
    pub set_layouts: SmallVec<[super::descriptor_set_layout::DescriptorSetLayout; 4]>,
    pub push_constants: SmallVec<[vk::PushConstantRange; 4]>,
}

impl PipelineLayoutCreateInfo {
    pub unsafe fn create(&self, ctx: &Device) -> VulkanResult<vk::PipelineLayout> {
        let mut layouts = self
            .set_layouts
            .iter()
            .map(|s| s.0.get_handle())
            .collect::<SmallVec<[_; 8]>>();

        let info = vk::PipelineLayoutCreateInfo {
            flags: self.flags,
            set_layout_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ffi_ptr(),
            push_constant_range_count: self.push_constants.len() as u32,
            p_push_constant_ranges: self.push_constants.as_ffi_ptr(),
            ..Default::default()
        };

        ctx.device()
            .create_pipeline_layout(&info, ctx.allocator_callbacks())
    }
}

create_object! {PipelineLayout}
impl Object for PipelineLayout {
    type Storage = SimpleStorage<Self>;
    type Parent = Device;

    type InputData = PipelineLayoutCreateInfo;
    type Data = BasicObjectData<vk::PipelineLayout, PipelineLayoutCreateInfo>;

    unsafe fn create(data: Self::InputData, ctx: &Self::Parent) -> VulkanResult<Self::Data> {
        BasicObjectData::new_result(data.create(ctx), data)
    }

    unsafe fn destroy(
        data: &Self::Data,
        lock: &SynchronizationLock,
        ctx: &Self::Parent,
    ) -> VulkanResult<()> {
        ctx.device
            .destroy_pipeline_layout(data.handle, ctx.allocator_callbacks());
        VulkanResult::Ok(())
    }

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        todo!()
    }
}
