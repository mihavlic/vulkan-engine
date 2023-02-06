use std::ptr;

use crate::device::Device;
use crate::storage::nostore::SimpleStorage;
use crate::storage::SynchronizationLock;
use crate::util::ffi_ptr::AsFFiPtr;

use super::BasicObjectData;
use super::{ObjHandle, Object};
use pumice::util::ObjectHandle;
use pumice::vk;
use pumice::VulkanResult;
use smallvec::SmallVec;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DescriptorBinding {
    pub binding: u32,
    pub count: u32,
    pub kind: vk::DescriptorType,
    pub stages: vk::ShaderStageFlags,
    pub immutable_samplers: SmallVec<[super::Sampler; 2]>,
}

impl Default for DescriptorBinding {
    fn default() -> Self {
        Self {
            binding: 0,
            count: 1,
            kind: Default::default(),
            stages: Default::default(),
            immutable_samplers: Default::default(),
        }
    }
}

impl DescriptorBinding {
    unsafe fn to_vk_multiple(
        this: &[Self],
        samplers: &mut SmallVec<[vk::Sampler; 8]>,
        out: &mut SmallVec<[vk::DescriptorSetLayoutBinding; 8]>,
    ) {
        // first collect all bindings, we cannot get stable poiners until everything is collected because a vector can reallocate
        for binding in this {
            if !binding.immutable_samplers.is_empty() {
                assert_eq!(binding.kind, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, "Immutable samplers are only supported with COMBINED_IMAGE_SAMPLER descriptor bindings");
                assert_eq!(
                    binding.count as usize,
                    binding.immutable_samplers.len(),
                    "CombinedImageSampler descriptor must have a corresponding number of samplers"
                );
                samplers.extend(binding.immutable_samplers.iter().map(|s| s.get_handle()))
            }
        }

        // now fill in the output structures
        let mut samplers_cursor = 0;
        out.extend(this.iter().map(|binding| {
            let ptr = if !binding.immutable_samplers.is_empty() {
                let ptr = samplers.as_ffi_ptr().add(binding.immutable_samplers.len());
                samplers_cursor += binding.immutable_samplers.len();
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

#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct DescriptorSetLayoutCreateInfo {
    pub flags: vk::DescriptorSetLayoutCreateFlags,
    pub bindings: Vec<DescriptorBinding>,
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

    fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.descriptor_set_layouts
    }
}
