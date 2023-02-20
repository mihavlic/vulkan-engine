use pumice::{vk, VulkanResult};
use smallvec::SmallVec;
use spirq::Variable;

use crate::object;

use super::Device;

#[derive(Debug)]
pub struct DescriptorSetBinding {
    pub name: String,
    pub binding: u32,
    pub count: u32,
    pub kind: vk::DescriptorType,
    pub stages: vk::ShaderStageFlags,
}

#[derive(Debug)]
pub struct DescriptorSet {
    pub set: u32,
    pub bindings: Vec<DescriptorSetBinding>,
}

#[derive(Debug)]
pub struct PushConstantRange {
    pub name: String,
    pub stage_flags: vk::ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug)]
pub struct ReflectedLayout {
    pub sets: Vec<DescriptorSet>,
    pub push_constants: Vec<PushConstantRange>,
}

pub struct SpirvModule<'a, 'b, 'c> {
    pub spirv: &'a [u32],
    pub entry_points: &'b [&'c str],
    pub dynamic_uniform_buffers: bool,
    pub dynamic_storage_buffers: bool,
    pub include_unused_descriptors: bool,
}

impl ReflectedLayout {
    /// spirv blob, entry points, dynamic buffers, reference unused desciptors
    pub fn new(modules: &[SpirvModule]) -> Self {
        let mut sets: Vec<DescriptorSet> = Vec::new();
        let mut push_constants: Vec<PushConstantRange> = Vec::new();
        let mut push_constant_cursor = 0;

        #[rustfmt::skip]
        fn type_to_descriptor_type(ty: spirq::DescriptorType, dynamic_uniform_buffers: bool, dynamic_storage_buffers: bool) -> vk::DescriptorType {
            match ty {
                spirq::DescriptorType::Sampler() => vk::DescriptorType::SAMPLER,
                spirq::DescriptorType::CombinedImageSampler() => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                spirq::DescriptorType::SampledImage() => vk::DescriptorType::SAMPLED_IMAGE,
                spirq::DescriptorType::StorageImage(_) => vk::DescriptorType::STORAGE_IMAGE,
                spirq::DescriptorType::UniformTexelBuffer() => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                spirq::DescriptorType::StorageTexelBuffer(_) => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                spirq::DescriptorType::UniformBuffer() if dynamic_uniform_buffers => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                spirq::DescriptorType::StorageBuffer(_) if dynamic_storage_buffers => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                spirq::DescriptorType::UniformBuffer() => vk::DescriptorType::UNIFORM_BUFFER,
                spirq::DescriptorType::StorageBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
                spirq::DescriptorType::InputAttachment(_) => vk::DescriptorType::INPUT_ATTACHMENT,
                // Provided by VK_KHR_acceleration_structure
                spirq::DescriptorType::AccelStruct() => vk::DescriptorType(1000150000),
            }
        }

        #[rustfmt::skip]
        fn exec_model_to_stages(model: spirq::ExecutionModel) -> vk::ShaderStageFlags {
            match model {
                spirq::ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
                spirq::ExecutionModel::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
                spirq::ExecutionModel::TessellationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                spirq::ExecutionModel::Geometry => vk::ShaderStageFlags::GEOMETRY,
                spirq::ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
                spirq::ExecutionModel::GLCompute => vk::ShaderStageFlags::COMPUTE,
                spirq::ExecutionModel::Kernel => panic!("ExecutionModel::Kernel is an OpenCL model"),
                // these come from extensions and since we are probably not generating them, the raw values have been inlined here
                spirq::ExecutionModel::TaskNV => vk::ShaderStageFlags(0x00000040),
                spirq::ExecutionModel::MeshNV => vk::ShaderStageFlags(0x00000080),
                spirq::ExecutionModel::RayGenerationNV => vk::ShaderStageFlags(0x00000100),
                spirq::ExecutionModel::IntersectionNV => vk::ShaderStageFlags(0x00001000),
                spirq::ExecutionModel::AnyHitNV => vk::ShaderStageFlags(0x00000200),
                spirq::ExecutionModel::ClosestHitNV => vk::ShaderStageFlags(0x00000400),
                spirq::ExecutionModel::MissNV => vk::ShaderStageFlags(0x00000800),
                spirq::ExecutionModel::CallableNV => vk::ShaderStageFlags(0x00002000),
            }
        }

        for &SpirvModule {
            spirv,
            entry_points,
            dynamic_uniform_buffers,
            dynamic_storage_buffers,
            include_unused_descriptors,
        } in modules
        {
            let reflected = spirq::ReflectConfig::new()
                .spv(spirv)
                .ref_all_rscs(include_unused_descriptors)
                .reflect()
                .unwrap();

            for entry in reflected {
                if entry_points.contains(&entry.name.as_str()) {
                    let stage = exec_model_to_stages(entry.exec_model);
                    for var in entry.vars {
                        match var {
                            Variable::Descriptor {
                                name,
                                desc_bind,
                                desc_ty,
                                ty,
                                nbind: count,
                            } => {
                                let mut name = name.unwrap_or_default();
                                let set_index = desc_bind.set();
                                let binding_index = desc_bind.bind();
                                let kind = type_to_descriptor_type(
                                    desc_ty,
                                    dynamic_uniform_buffers,
                                    dynamic_storage_buffers,
                                );

                                let mut new_binding = || DescriptorSetBinding {
                                    name: std::mem::take(&mut name),
                                    binding: desc_bind.bind(),
                                    count,
                                    kind,
                                    stages: stage,
                                };
                                if let Some(set) = sets.iter_mut().find(|set| set.set == set_index)
                                {
                                    if let Some(binding) =
                                        set.bindings.iter_mut().find(|b| b.binding == binding_index)
                                    {
                                        assert_eq!(binding.kind, kind, "Two entries for binding #{binding_index} of set #{set_index} have a different descriptor type: {:?} vs {:?} (named {} and {})", binding.kind, kind, binding.name, name);
                                        binding.stages |= stage;
                                        binding.count = binding.count.max(count);
                                    } else {
                                        set.bindings.push(new_binding());
                                    }
                                } else {
                                    sets.push(DescriptorSet {
                                        set: set_index,
                                        bindings: vec![new_binding()],
                                    });
                                }
                            }
                            Variable::PushConstant { name, ty } => {
                                let name = name.unwrap_or_default();
                                let size: u32 = ty.nbyte().unwrap().try_into().unwrap();
                                if let Some(found) =
                                    push_constants.iter_mut().find(|entry| entry.name == name)
                                {
                                    assert_eq!(found.size, size, "Push constant ranges named '{name}' occur multiple times with different sizes");
                                    found.stage_flags |= stage;
                                } else {
                                    push_constants.push(PushConstantRange {
                                        name,
                                        stage_flags: stage,
                                        offset: push_constant_cursor,
                                        size,
                                    });

                                    push_constant_cursor += size;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        sets.sort_by_key(|set| set.set);
        for set in &mut sets {
            set.bindings.sort_by_key(|b| b.binding);
        }

        ReflectedLayout {
            sets,
            push_constants,
        }
    }
    pub unsafe fn create(
        &self,
        device: &Device,
        set_layout_flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> VulkanResult<object::PipelineLayout> {
        self.create_with_samplers(device, set_layout_flags, |_, _| None)
    }
    pub unsafe fn create_with_samplers<
        F: FnMut(&DescriptorSet, &DescriptorSetBinding) -> Option<SmallVec<[object::Sampler; 2]>>,
    >(
        &self,
        device: &Device,
        set_layout_flags: vk::DescriptorSetLayoutCreateFlags,
        sampler_fun: F,
    ) -> VulkanResult<object::PipelineLayout> {
        let layouts = self.create_descriptor_set_layouts_with_samplers(
            set_layout_flags,
            device,
            sampler_fun,
        )?;

        let push_constants = self
            .push_constants
            .iter()
            .map(|p| vk::PushConstantRange {
                stage_flags: p.stage_flags,
                offset: p.offset,
                size: p.size,
            })
            .collect();

        let info = object::PipelineLayoutCreateInfo {
            set_layouts: layouts,
            push_constants,
        };
        device.create_pipeline_layout(info)
    }
    pub unsafe fn create_descriptor_set_layouts(
        &self,
        device: &Device,
        set_layout_flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> VulkanResult<Vec<object::DescriptorSetLayout>> {
        self.create_descriptor_set_layouts_with_samplers(set_layout_flags, device, |_, _| None)
    }
    pub unsafe fn create_descriptor_set_layouts_with_samplers<
        F: FnMut(&DescriptorSet, &DescriptorSetBinding) -> Option<SmallVec<[object::Sampler; 2]>>,
    >(
        &self,
        set_layout_flags: vk::DescriptorSetLayoutCreateFlags,
        device: &Device,
        mut sampler_fun: F,
    ) -> VulkanResult<Vec<object::DescriptorSetLayout>> {
        let layouts = self
            .sets
            .iter()
            .map(|set| {
                let bindings = set
                    .bindings
                    .iter()
                    .map(|b| object::DescriptorBinding {
                        binding: b.binding,
                        count: b.count,
                        kind: b.kind,
                        stages: b.stages,
                        immutable_samplers: sampler_fun(set, b).unwrap_or_default(),
                    })
                    .collect();
                let info = object::DescriptorSetLayoutCreateInfo {
                    flags: set_layout_flags,
                    bindings,
                };
                device.create_descriptor_set_layout(info)
            })
            .collect::<VulkanResult<Vec<_>>>()?;
        Ok(layouts)
    }
}
