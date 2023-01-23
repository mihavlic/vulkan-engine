use std::cell::RefMut;
use std::ptr;

use super::{ArcHandle, BasicObjectData, ImageMutableState, Object, ObjectData};

use crate::device::Device;
use crate::storage::nostore::SimpleStorage;
use crate::storage::{MutableShared, SynchronizationLock};
use pumice::util::ObjectHandle;
use pumice::vk;
use pumice::VulkanResult;

create_object! {ShaderModule}
impl Object for ShaderModule {
    type Storage = SimpleStorage<Self>;
    type Parent = Device;

    type InputData<'a> = &'a [u32];
    type Data = BasicObjectData<vk::ShaderModule, ()>;

    unsafe fn create<'a>(
        data: Self::InputData<'a>,
        ctx: &Self::Parent,
    ) -> VulkanResult<Self::Data> {
        let create_info = vk::ShaderModuleCreateInfo {
            code_size: data.len() * 4,
            p_code: data.as_ptr(),
            ..Default::default()
        };

        let handle = ctx
            .device()
            .create_shader_module(&create_info, ctx.allocator_callbacks());

        BasicObjectData::new_result(handle, ())
    }

    unsafe fn destroy(
        data: &Self::Data,
        lock: &SynchronizationLock,
        ctx: &Self::Parent,
    ) -> VulkanResult<()> {
        ctx.device()
            .destroy_shader_module(data.get_handle(), ctx.allocator_callbacks());

        VulkanResult::Ok(())
    }

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.shader_modules
    }
}
