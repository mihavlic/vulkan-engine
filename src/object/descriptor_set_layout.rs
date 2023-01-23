use std::ptr;

use crate::device::Device;
use crate::storage::nostore::SimpleStorage;
use crate::storage::SynchronizationLock;
use crate::util::ffi_ptr::AsFFiPtr;

use super::BasicObjectData;
use super::{ArcHandle, Object};
use pumice::util::ObjectHandle;
use pumice::vk;
use pumice::VulkanResult;
use smallvec::SmallVec;

#[derive(Clone, PartialEq, Eq, Hash)]
struct InnerSampler;

impl InnerSampler {
    fn handle(&self) -> vk::Sampler {
        todo!()
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Sampler(InnerSampler);

// #[derive(Clone, PartialEq, Eq, Hash)]
// pub enum DescriptorKind {
//     Sampler,
//     CombinedImageSampler(Sampler),
//     SampledImage,
//     StorageImage,
//     UniformTexelBuffer,
//     StorageTexelBuffer,
//     UniformBuffer,
//     StorageBuffer,
//     UniformBufferDynamic,
//     StorageBufferDynamic,
//     InputAttachment, // TODO support other values
// }

// impl DescriptorKind {
//     fn to_vk(&self) -> (vk::DescriptorType, Option<vk::Sampler>) {
//         use DescriptorKind::*;
//         let mut sampler = None;
//         let kind = match self {
//             Sampler => vk::DescriptorType::SAMPLER,
//             CombinedImageSampler(combined_sampler) => {
//                 sampler = Some(combined_sampler.0.handle());
//                 vk::DescriptorType::COMBINED_IMAGE_SAMPLER
//             }
//             SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
//             StorageImage => vk::DescriptorType::STORAGE_IMAGE,
//             UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
//             StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
//             UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
//             StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
//             UniformBufferDynamic => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
//             StorageBufferDynamic => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
//             InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
//         };

//         (kind, sampler)
//     }
// }

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DescriptorBinding {
    binding: u32,
    count: u32,
    kind: vk::DescriptorType,
    stages: vk::ShaderStageFlags,
    immutable_samplers: Option<SmallVec<[Sampler; 2]>>,
}

impl DescriptorBinding {
    unsafe fn to_vk_multiple(
        this: &[Self],
        samplers: &mut SmallVec<[vk::Sampler; 8]>,
        out: &mut SmallVec<[vk::DescriptorSetLayoutBinding; 8]>,
    ) {
        // first collect all bindings, we cannot get stable poiners until everything is collected because a vector can reallocate
        for binding in this {
            if let Some(immutable_samplers) = &binding.immutable_samplers {
                assert_eq!(binding.kind, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, "Immutable samplers are only supported with COMBINED_IMAGE_SAMPLER descriptor bindings");
                assert_eq!(
                    binding.count as usize,
                    immutable_samplers.len(),
                    "CombinedImageSampler descriptor must have a corresponding number of samplers"
                );
                samplers.extend(immutable_samplers.iter().map(|s| s.0.handle()))
            }
        }

        // now fill in the output structures
        let mut samplers_cursor = 0;
        out.extend(this.iter().map(|binding| {
            let ptr = if let Some(immutable_samplers) = &binding.immutable_samplers {
                let ptr = samplers.as_ffi_ptr().add(immutable_samplers.len());
                samplers_cursor += immutable_samplers.len();
                ptr
            } else {
                std::ptr::null()
            };

            vk::DescriptorSetLayoutBinding {
                binding: binding.binding,
                descriptor_type: binding.kind,
                descriptor_count: binding.count,
                stage_flags: binding.stages,
                p_immutable_samplers: ptr,
            }
        }));
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DescriptorSetLayoutCreateInfo {
    flags: vk::DescriptorSetLayoutCreateFlags,
    bindings: Vec<DescriptorBinding>,
}

impl DescriptorSetLayoutCreateInfo {
    pub unsafe fn create(&self, device: &Device) -> VulkanResult<vk::DescriptorSetLayout> {
        let mut samplers = SmallVec::new();
        let mut bindings = SmallVec::new();
        DescriptorBinding::to_vk_multiple(&self.bindings, &mut samplers, &mut bindings);

        let info = vk::DescriptorSetLayoutCreateInfo {
            flags: self.flags,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ffi_ptr(),
            ..Default::default()
        };

        device
            .device()
            .create_descriptor_set_layout(&info, device.allocator_callbacks())
    }
}

create_object! {DescriptorSetLayout}
derive_raw_handle! {DescriptorSetLayout, vk::DescriptorSetLayout}
impl Object for DescriptorSetLayout {
    type Storage = SimpleStorage<Self>;
    type Parent = Device;

    type InputData<'a> = DescriptorSetLayoutCreateInfo;
    type Data = BasicObjectData<vk::DescriptorSetLayout, DescriptorSetLayoutCreateInfo>;

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
            .destroy_descriptor_set_layout(data.handle, ctx.allocator_callbacks());
        VulkanResult::Ok(())
    }

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.descriptor_set_layouts
    }
}
