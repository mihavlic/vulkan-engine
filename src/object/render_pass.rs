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

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SubpassDescription {
    pub flags: vk::SubpassDescriptionFlags,
    pub pipeline_bind_point: vk::PipelineBindPoint,
    pub input_attachments: Vec<vk::AttachmentReference>,
    pub color_attachments: Vec<vk::AttachmentReference>,
    pub resolve_attachments: Vec<vk::AttachmentReference>,
    pub depth_stencil_attachment: Option<vk::AttachmentReference>,
    pub preserve_attachments: Vec<u32>,
}
impl SubpassDescription {
    pub fn to_vk(&self) -> vk::SubpassDescription {
        vk::SubpassDescription {
            flags: self.flags,
            pipeline_bind_point: self.pipeline_bind_point,
            input_attachment_count: self.input_attachments.len() as u32,
            p_input_attachments: self.input_attachments.as_ffi_ptr(),
            color_attachment_count: self.color_attachments.len() as u32,
            p_color_attachments: self.color_attachments.as_ffi_ptr(),
            p_resolve_attachments: self.resolve_attachments.as_ffi_ptr(),
            p_depth_stencil_attachment: self.depth_stencil_attachment.as_ffi_ptr(),
            preserve_attachment_count: self.preserve_attachments.len() as u32,
            p_preserve_attachments: self.preserve_attachments.as_ffi_ptr(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RenderPassCreateInfo {
    pub flags: vk::RenderPassCreateFlags,
    pub attachments: Vec<vk::AttachmentDescription>,
    pub subpasses: Vec<SubpassDescription>,
    pub dependencies: Vec<vk::SubpassDependency>,
}

impl RenderPassCreateInfo {
    pub unsafe fn create(&self, ctx: &Device) -> VulkanResult<vk::RenderPass> {
        let mut subpasses = self
            .subpasses
            .iter()
            .map(|s| s.to_vk())
            .collect::<SmallVec<[_; 8]>>();

        let info = vk::RenderPassCreateInfo {
            flags: todo!(),
            attachment_count: self.attachments.len() as u32,
            p_attachments: self.attachments.as_ffi_ptr(),
            subpass_count: subpasses.len() as u32,
            p_subpasses: subpasses.as_ffi_ptr(),
            dependency_count: self.dependencies.len() as u32,
            p_dependencies: self.dependencies.as_ffi_ptr(),
            ..Default::default()
        };

        ctx.device()
            .create_render_pass(&info, ctx.allocator_callbacks())
    }
}

create_object! {RenderPass}
derive_raw_handle! {RenderPass, vk::RenderPass}
impl Object for RenderPass {
    type Storage = SimpleStorage<Self>;
    type Parent = Device;

    type InputData<'a> = RenderPassCreateInfo;
    type Data = BasicObjectData<vk::RenderPass, RenderPassCreateInfo>;

    unsafe fn create<'a>(
        data: Self::InputData<'a>,
        ctx: &Self::Parent,
    ) -> VulkanResult<Self::Data> {
        BasicObjectData::new_result(data.create(ctx), data)
    }

    unsafe fn destroy(
        data: &Self::Data,
        lock: &SynchronizationLock,
        ctx: &Self::Parent,
    ) -> VulkanResult<()> {
        ctx.device
            .destroy_render_pass(data.handle, ctx.allocator_callbacks());
        VulkanResult::Ok(())
    }

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.render_passes
    }
}
