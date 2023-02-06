use std::hash::{Hash, Hasher};
use std::ptr;

use crate::device::Device;
use crate::storage::interned::ObjectCreateInfoFingerPrint;
use crate::storage::nostore::SimpleStorage;
use crate::storage::{constant_ahash_hasher, SynchronizationLock};
use crate::util::ffi_ptr::AsFFiPtr;

use super::{BasicObjectData, ObjectData};
use super::{ObjHandle, Object};
use pumice::util::ObjectHandle;
use pumice::vk;
use pumice::VulkanResult;
use smallvec::SmallVec;

#[derive(Clone, Default, PartialEq)]
pub struct SamplerCreateInfo {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub mip_lod_bias: f32,
    pub anisotropy_enable: bool,
    pub max_anisotropy: f32,
    pub compare_enable: bool,
    pub compare_op: vk::CompareOp,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: vk::BorderColor,
    pub unnormalized_coordinates: bool,
}

impl SamplerCreateInfo {
    pub unsafe fn to_vk(&self) -> vk::SamplerCreateInfo {
        vk::SamplerCreateInfo {
            mag_filter: self.mag_filter,
            min_filter: self.min_filter,
            mipmap_mode: self.mipmap_mode,
            address_mode_u: self.address_mode_u,
            address_mode_v: self.address_mode_v,
            address_mode_w: self.address_mode_w,
            mip_lod_bias: self.mip_lod_bias,
            anisotropy_enable: self.anisotropy_enable as vk::Bool32,
            max_anisotropy: self.max_anisotropy,
            compare_enable: self.compare_enable as vk::Bool32,
            compare_op: self.compare_op,
            min_lod: self.min_lod,
            max_lod: self.max_lod,
            border_color: self.border_color,
            unnormalized_coordinates: self.unnormalized_coordinates as vk::Bool32,
            ..Default::default()
        }
    }
}

create_object! {Sampler}
derive_raw_handle! {Sampler, vk::Sampler}
impl Object for Sampler {
    type Storage = SimpleStorage<Self>;
    type Parent = Device;

    type InputData<'a> = SamplerCreateInfo;
    type Data = BasicObjectData<vk::Sampler, SamplerCreateInfo>;

    unsafe fn create<'a>(
        data: Self::InputData<'a>,
        ctx: &Self::Parent,
    ) -> VulkanResult<Self::Data> {
        let create_info = data.to_vk();
        BasicObjectData::new_result(
            ctx.device()
                .create_sampler(&create_info, ctx.allocator_callbacks()),
            data,
        )
    }

    unsafe fn destroy(
        data: &Self::Data,
        lock: &SynchronizationLock,
        ctx: &Self::Parent,
    ) -> VulkanResult<()> {
        ctx.device()
            .destroy_sampler(data.get_handle(), ctx.allocator_callbacks());
        VulkanResult::Ok(())
    }

    fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.samplers
    }
}
