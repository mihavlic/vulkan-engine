use std::hash::{Hash, Hasher};
use std::ptr;

use crate::device::Device;
use crate::storage::interned::ObjectCreateInfoFingerPrint;
use crate::storage::nostore::SimpleStorage;
use crate::storage::{constant_ahash_hasher, SynchronizationLock};
use crate::util::ffi_ptr::AsFFiPtr;

use super::{BasicObjectData, ObjRef, ObjectData};
use super::{ObjHandle, Object};
use bumpalo::Bump;
use pumice::util::ObjectHandle;
use pumice::vk;
use pumice::VulkanResult;
use smallvec::{smallvec, SmallVec};

#[derive(Clone)]
pub struct ImagelessAttachment {
    pub flags: vk::ImageCreateFlags,
    pub usage: vk::ImageUsageFlags,
    pub width: u32,
    pub height: u32,
    pub layer_count: u32,
    pub view_formats: SmallVec<[vk::Format; 2]>,
}

impl ImagelessAttachment {
    pub fn from_image(image: &ObjRef<super::Image>) -> Self {
        let info = image.get_create_info();
        let (width, height) = info.size.get_2d().unwrap();
        Self {
            flags: info.flags,
            usage: info.usage,
            width,
            height,
            layer_count: info.array_layers,
            view_formats: smallvec![info.format],
        }
    }
    pub fn to_vk(&self) -> vk::FramebufferAttachmentImageInfoKHR {
        vk::FramebufferAttachmentImageInfoKHR {
            flags: self.flags,
            usage: self.usage,
            width: self.width,
            height: self.height,
            layer_count: self.layer_count,
            view_format_count: self.view_formats.len() as u32,
            p_view_formats: self.view_formats.as_ffi_ptr(),
            ..Default::default()
        }
    }
}

#[derive(Clone)]
pub enum AttachmentMode {
    Normal(Vec<(super::Image, vk::ImageView)>),
    Imageless(Vec<ImagelessAttachment>),
}

impl Default for AttachmentMode {
    fn default() -> Self {
        Self::Normal(Vec::new())
    }
}

impl AttachmentMode {
    pub fn imageless_from_images<'a>(
        images: impl IntoIterator<Item = &'a ObjRef<super::Image>>,
    ) -> Self {
        Self::Imageless(
            images
                .into_iter()
                .map(ImagelessAttachment::from_image)
                .collect(),
        )
    }
}

#[derive(Clone)]
pub struct FramebufferCreateInfo {
    pub flags: vk::FramebufferCreateFlags,
    pub render_pass: super::RenderPass,
    pub attachments: AttachmentMode,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
}

impl FramebufferCreateInfo {
    pub unsafe fn create(&self, ctx: &Device) -> VulkanResult<vk::Framebuffer> {
        // FIXME do better
        let bump = Bump::new();
        let mut pnext_head: *const std::ffi::c_void = std::ptr::null();

        let mut flags = self.flags;

        let attachments = match &self.attachments {
            AttachmentMode::Normal(s) => {
                &*bump.alloc_slice_fill_iter(s.iter().map(|&(_, view)| view))
            }
            AttachmentMode::Imageless(s) => {
                flags |= vk::FramebufferCreateFlags::IMAGELESS;

                let imageless =
                    bump.alloc_slice_fill_iter(s.iter().map(ImagelessAttachment::to_vk));
                let p = bump.alloc(vk::FramebufferAttachmentsCreateInfoKHR {
                    attachment_image_info_count: imageless.len() as u32,
                    p_attachment_image_infos: imageless.as_ffi_ptr(),
                    ..Default::default()
                });
                add_pnext!(pnext_head, p);

                &[]
            }
        };

        let info = vk::FramebufferCreateInfo {
            p_next: pnext_head,
            flags,
            render_pass: self.render_pass.get_handle(),
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ffi_ptr(),
            width: self.width,
            height: self.height,
            layers: self.layers,
            ..Default::default()
        };

        ctx.device()
            .create_framebuffer(&info, ctx.allocator_callbacks())
    }
}

create_object! {Framebuffer}
derive_raw_handle! {Framebuffer, vk::Framebuffer}
impl Object for Framebuffer {
    type Storage = SimpleStorage<Self>;
    type Parent = Device;

    type InputData<'a> = FramebufferCreateInfo;
    type Data = BasicObjectData<vk::Framebuffer, FramebufferCreateInfo>;

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
            .destroy_framebuffer(data.handle, ctx.allocator_callbacks());
        VulkanResult::Ok(())
    }

    fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.framebuffers
    }
}
