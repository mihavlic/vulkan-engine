use std::hash::{Hash, Hasher};
use std::ptr;

use crate::device::Device;
use crate::storage::interned::ObjectCreateInfoFingerPrint;
use crate::storage::nostore::SimpleStorage;
use crate::storage::{constant_ahash_hasher, SynchronizationLock};
use crate::util::ffi_ptr::AsFFiPtr;

use super::{BasePipeline, BasicObjectData, ObjectData};
use super::{ObjHandle, Object};
use bumpalo::Bump;
use pumice::util::ObjectHandle;
use pumice::vk;
use pumice::VulkanResult;
use smallvec::SmallVec;

#[derive(Clone, Default)]
pub struct ComputePipelineCreateInfoBuilder {
    pub flags: vk::PipelineCreateFlags,
    pub stage: Option<super::PipelineStage>,
    pub layout: Option<super::PipelineLayout>,
    pub base_pipeline: super::BasePipeline,
}

impl ComputePipelineCreateInfoBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn flags(mut self, flags: vk::PipelineCreateFlags) -> Self {
        self.flags = flags;
        self
    }
    pub fn stage(mut self, stage: super::PipelineStage) -> Self {
        self.stage = Some(stage);
        self
    }
    pub fn layout(mut self, layout: super::PipelineLayout) -> Self {
        self.layout = Some(layout);
        self
    }
    pub fn base_pipeline(mut self, base_pipeline: super::BasePipeline) -> Self {
        self.base_pipeline = base_pipeline;
        self
    }
    pub fn finish(self) -> ComputePipelineCreateInfo {
        ComputePipelineCreateInfo {
            flags: self.flags,
            stage: self.stage.expect("Pipeline must have stage provided"),
            layout: self.layout.expect("Pipeline must have layout provided"),
            base_pipeline: self.base_pipeline,
        }
    }
}

#[derive(Clone)]
pub struct ComputePipelineCreateInfo {
    pub flags: vk::PipelineCreateFlags,
    pub stage: super::PipelineStage,
    pub layout: super::PipelineLayout,
    pub base_pipeline: super::BasePipeline,
}

impl ComputePipelineCreateInfo {
    pub fn builder() -> ComputePipelineCreateInfoBuilder {
        ComputePipelineCreateInfoBuilder::default()
    }
    pub unsafe fn to_vk(&self, bump: &Bump) -> vk::ComputePipelineCreateInfo {
        let (base_pipeline_handle, base_pipeline_index) =
            self.base_pipeline.get_index_or_pipeline();

        vk::ComputePipelineCreateInfo {
            flags: self.flags,
            stage: self.stage.to_vk(bump),
            layout: self.layout.get_handle(),
            base_pipeline_handle,
            base_pipeline_index,
            ..Default::default()
        }
    }
}

create_object! {ComputePipeline}
derive_raw_handle! {ComputePipeline, vk::Pipeline}
impl Object for ComputePipeline {
    type Storage = SimpleStorage<Self>;
    type Parent = Device;

    type InputData<'a> = ComputePipelineCreateInfo;
    type Data = BasicObjectData<vk::Pipeline, ComputePipelineCreateInfo>;

    unsafe fn create<'a>(
        data: Self::InputData<'a>,
        ctx: &Self::Parent,
    ) -> VulkanResult<Self::Data> {
        let bump = Bump::new();
        create_compute_pipeline_impl(data, &bump, ctx)
    }

    unsafe fn destroy(
        data: &Self::Data,
        lock: &SynchronizationLock,
        ctx: &Self::Parent,
    ) -> VulkanResult<()> {
        ctx.device()
            .destroy_pipeline(data.get_handle(), ctx.allocator_callbacks());
        VulkanResult::Ok(())
    }

    fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        &parent.compute_pipelines
    }
}

impl ComputePipeline {
    pub fn get_descriptor_set_layouts(&self) -> &Vec<super::DescriptorSetLayout> {
        self.get_create_info().layout.get_descriptor_set_layouts()
    }
    pub fn get_pipeline_layout(&self) -> &super::PipelineLayout {
        &self.get_create_info().layout
    }
}

pub(crate) unsafe fn create_compute_pipeline_impl(
    data: ComputePipelineCreateInfo,
    bump: &Bump,
    ctx: &Device,
) -> VulkanResult<BasicObjectData<vk::Pipeline, ComputePipelineCreateInfo>> {
    let create_info = data.to_vk(&bump);
    let (pipelines, result) = ctx.device().create_compute_pipelines(
        ctx.pipeline_cache(),
        &[create_info],
        ctx.allocator_callbacks(),
    )?;
    assert_eq!(result, vk::Result::SUCCESS);
    BasicObjectData::new(pipelines[0], data)
}
