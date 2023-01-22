use std::{
    borrow::Cow,
    hash::{Hash, Hasher},
    ptr,
    sync::Arc,
};

use crate::{
    arena::uint::OptionalU32,
    device::{batch::GenerationId, Device},
    storage::{constant_ahash_hasher, nostore::SimpleStorage, MutableShared, SynchronizationLock},
    util::ffi_ptr::AsFFiPtr,
};
use parking_lot::{RawMutex, RawRwLock};
use pumice::{util::ObjectHandle, vk, VulkanResult};
use smallvec::SmallVec;

use super::{ArcHandle, BasicObjectData, Object, ObjectData};

#[derive(Clone)]
pub struct SpecializationMapEntry {
    pub constant_id: u32,
    pub offset: u32,
    pub size: usize,
}

#[derive(Clone)]
pub struct SpecializationInfo {
    pub map_entries: Vec<SpecializationMapEntry>,
    pub data: Vec<u8>,
}

#[derive(Clone)]
pub struct PipelineShaderStageCreateInfo {
    pub flags: vk::PipelineShaderStageCreateFlags,
    pub stage: vk::ShaderStageFlags,
    pub module: vk::ShaderModule,
    pub name: Cow<'static, str>,
    pub specialization_info: Option<SpecializationInfo>,
}

impl PipelineShaderStageCreateInfo {
    fn to_vk(&self) -> vk::PipelineShaderStageCreateInfo {
        vk::PipelineShaderStageCreateInfo {
            flags: self.flags,
            stage: self.stage,
            module: self.module,
            p_name: self.name.as_ptr() as *const std::ffi::c_char,
            p_specialization_info: todo!(),
            ..Default::default()
        }
    }
}

pub struct PipelineRenderingCreateInfoKHR {
    pub s_type: vk::StructureType,
    pub p_next: *const std::os::raw::c_void,
    pub view_mask: u32,
    pub color_attachment_count: u32,
    pub p_color_attachment_formats: *const vk::Format,
    pub depth_attachment_format: vk::Format,
    pub stencil_attachment_format: vk::Format,
}

#[derive(Clone)]
pub enum RenderPassMode {
    Normal {
        subpass: u32,
        render_pass: super::RenderPass,
    },
    Dynamic {
        view_mask: u32,
        colors: SmallVec<[vk::Format; 4]>,
        depth: vk::Format,
        stencil: vk::Format,
    },
}

impl RenderPassMode {
    pub fn get_hash(&self) -> u64 {
        let mut hasher = constant_ahash_hasher();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

impl Hash for RenderPassMode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            RenderPassMode::Normal {
                subpass,
                render_pass,
            } => {
                subpass.hash(state);
                render_pass.0.hash(state);
            }
            RenderPassMode::Dynamic {
                view_mask,
                colors,
                depth,
                stencil,
            } => {
                view_mask.hash(state);
                colors.hash(state);
                depth.hash(state);
                stencil.hash(state);
            }
        }
    }
}

#[derive(Clone)]
pub enum BasePipeline {
    Index(i32),
    Pipeline(ConcreteGraphicsPipeline),
    None,
}

#[derive(Clone)]
pub struct GraphicsPipelineCreateInfo {
    pub flags: vk::PipelineCreateFlags,
    pub stages: Vec<PipelineShaderStageCreateInfo>,

    pub vertex_input_state: vk::PipelineVertexInputStateCreateInfo,
    pub input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo,
    pub tessellation_state: vk::PipelineTessellationStateCreateInfo,
    pub viewport_state: vk::PipelineViewportStateCreateInfo,
    pub rasterization_state: vk::PipelineRasterizationStateCreateInfo,
    pub multisample_state: vk::PipelineMultisampleStateCreateInfo,
    pub depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo,
    pub color_blend_state: vk::PipelineColorBlendStateCreateInfo,
    pub dynamic_state: Option<vk::PipelineDynamicStateCreateInfo>,

    pub layout: super::pipeline_layout::PipelineLayout,
    pub render_pass: RenderPassMode,
    pub base_pipeline: BasePipeline,
}

impl GraphicsPipelineCreateInfo {
    pub unsafe fn create_multiple<I: IntoIterator<Item = Self> + Clone>(
        this: I,
        pipeline_cache: vk::PipelineCache,
        create_infos: &mut SmallVec<[vk::GraphicsPipelineCreateInfo; 8]>,
        stages: &mut SmallVec<[vk::PipelineShaderStageCreateInfo; 8]>,
        dynamic_rendering_infos: &mut SmallVec<[vk::PipelineRenderingCreateInfoKHR; 8]>,
        ctx: &Device,
    ) -> VulkanResult<(Vec<vk::Pipeline>, vk::Result)> {
        let mut empty = true;

        for pipeline in this.clone() {
            empty = true;
            match pipeline.render_pass {
                RenderPassMode::Normal { .. } => {}
                RenderPassMode::Dynamic {
                    view_mask,
                    ref colors,
                    depth,
                    stencil,
                } => {
                    dynamic_rendering_infos.push(vk::PipelineRenderingCreateInfoKHR {
                        view_mask,
                        color_attachment_count: colors.len() as u32,
                        p_color_attachment_formats: colors.as_ffi_ptr(),
                        depth_attachment_format: depth,
                        stencil_attachment_format: stencil,
                        ..Default::default()
                    });
                }
            }
        }

        assert!(!empty);

        let mut rendering_info_offset = 0;
        let mut stages_offset = 0;
        for pipeline in this {
            let mut pnext_head: *const std::ffi::c_void = std::ptr::null();

            macro_rules! add_pnext {
                ($head:expr) => {
                    let ptr = &mut $head;
                    ptr.p_next = pnext_head;
                    pnext_head = ptr as *const _ as *const std::ffi::c_void;
                };
            }

            if let RenderPassMode::Dynamic { .. } = pipeline.render_pass {
                add_pnext!(dynamic_rendering_infos[rendering_info_offset]);
                rendering_info_offset += 1;
            };

            let stages_ptr = {
                let ptr = stages.as_ffi_ptr().add(stages_offset);
                stages_offset += pipeline.stages.len();
                ptr
            };

            let (render_pass, subpass) = match &pipeline.render_pass {
                RenderPassMode::Normal {
                    render_pass,
                    subpass,
                } => (render_pass.0.get_handle(), *subpass),
                RenderPassMode::Dynamic { .. } => (vk::RenderPass::null(), 0),
            };

            let (base_pipeline_handle, base_pipeline_index) = match &pipeline.base_pipeline {
                BasePipeline::Index(index) => (vk::Pipeline::null(), *index),
                BasePipeline::Pipeline(handle) => (handle.get_handle(), -1),
                BasePipeline::None => (vk::Pipeline::null(), -1),
            };

            let info = vk::GraphicsPipelineCreateInfo {
                p_next: pnext_head,
                flags: pipeline.flags,
                stage_count: pipeline.stages.len() as u32,
                p_stages: stages_ptr,
                p_vertex_input_state: &pipeline.vertex_input_state,
                p_input_assembly_state: &pipeline.input_assembly_state,
                p_tessellation_state: &pipeline.tessellation_state,
                p_viewport_state: &pipeline.viewport_state,
                p_rasterization_state: &pipeline.rasterization_state,
                p_multisample_state: &pipeline.multisample_state,
                p_depth_stencil_state: &pipeline.depth_stencil_state,
                p_color_blend_state: &pipeline.color_blend_state,
                p_dynamic_state: pipeline.dynamic_state.as_ffi_ptr(),
                layout: pipeline.layout.0.get_handle(),
                render_pass,
                subpass,
                base_pipeline_handle,
                base_pipeline_index,
                ..Default::default()
            };

            create_infos.push(info);
        }

        ctx.device().create_graphics_pipelines(
            pipeline_cache,
            &create_infos,
            ctx.allocator_callbacks(),
        )
    }
}

enum GraphicsPipelineEntryHandle {
    Promised(Arc<RawRwLock>),
    Created(vk::Pipeline),
}

pub(crate) struct GraphicsPipelineEntry {
    handle: GraphicsPipelineEntryHandle,
    mode_hash: u64,
    // TODO support pipeline garbage collection
    // last_use: GenerationId,
}

pub enum GetPipelineResult {
    Ready(vk::Pipeline),
    Promised(Arc<RawRwLock>),
    MustCreate(Arc<RawRwLock>),
}

pub struct GraphicsPipelineMutableState {
    pipelines: SmallVec<[GraphicsPipelineEntry; 2]>,
}

impl GraphicsPipelineMutableState {
    pub(crate) fn new() -> Self {
        Self {
            pipelines: SmallVec::new(),
        }
    }
    /// make_lock is a locked RwLock that unlocks for reading when the pipeline is created and inserted
    pub(crate) unsafe fn get_pipeline(
        &mut self,
        mode_hash: u64,
        make_lock: impl FnOnce() -> Arc<RawRwLock>,
    ) -> GetPipelineResult {
        if let Some(found) = self.pipelines.iter().find(|e| e.mode_hash == mode_hash) {
            match &found.handle {
                GraphicsPipelineEntryHandle::Promised(mutex) => {
                    GetPipelineResult::Promised(mutex.clone())
                }
                GraphicsPipelineEntryHandle::Created(ok) => GetPipelineResult::Ready(*ok),
            }
        } else {
            let lock = make_lock();

            self.pipelines.push(GraphicsPipelineEntry {
                handle: GraphicsPipelineEntryHandle::Promised(lock.clone()),
                mode_hash,
            });

            GetPipelineResult::MustCreate(lock)
        }
    }
    pub(crate) unsafe fn add_promised_pipeline(&mut self, handle: vk::Pipeline, mode_hash: u64) {
        if let Some(found) = self.pipelines.iter_mut().find(|e| e.mode_hash == mode_hash) {
            match &found.handle {
                GraphicsPipelineEntryHandle::Promised(mutex) => {
                    found.handle = GraphicsPipelineEntryHandle::Created(handle);
                }
                GraphicsPipelineEntryHandle::Created(_) => {
                    panic!("Promised pipeline already created!")
                }
            }
        } else {
            panic!("Entry missing for promised pipeline!");
        }
    }
    pub unsafe fn destroy(&mut self, ctx: &Device) {
        for entry in self.pipelines.drain(..) {
            match entry.handle {
                GraphicsPipelineEntryHandle::Promised(_) => {
                    panic!("Promised pipeline still pending during destruction")
                }
                GraphicsPipelineEntryHandle::Created(handle) => {
                    ctx.device()
                        .destroy_pipeline(handle, ctx.allocator_callbacks());
                }
            }
        }
    }
}

pub(crate) struct GraphicsPipelineState {
    info: GraphicsPipelineCreateInfo,
    pub(crate) mutable: MutableShared<GraphicsPipelineMutableState>,
}

impl ObjectData for GraphicsPipelineState {
    type CreateInfo = GraphicsPipelineCreateInfo;
    type Handle = ();

    fn get_create_info(&self) -> &Self::CreateInfo {
        &self.info
    }
    fn get_handle(&self) -> Self::Handle {
        ()
    }
}

create_object! {GraphicsPipeline}
impl Object for GraphicsPipeline {
    type Storage = SimpleStorage<Self>;
    type Parent = Device;

    type InputData = GraphicsPipelineCreateInfo;
    type Data = GraphicsPipelineState;

    unsafe fn create(data: Self::InputData, ctx: &Self::Parent) -> VulkanResult<Self::Data> {
        Ok(GraphicsPipelineState {
            info: data,
            mutable: MutableShared::new(GraphicsPipelineMutableState::new()),
        })
    }

    unsafe fn destroy(
        data: &Self::Data,
        lock: &SynchronizationLock,
        ctx: &Self::Parent,
    ) -> VulkanResult<()> {
        data.mutable.get_mut(lock).destroy(ctx);
        VulkanResult::Ok(())
    }

    unsafe fn get_storage(parent: &Self::Parent) -> &Self::Storage {
        todo!()
    }
}

#[derive(Clone)]
pub struct ConcreteGraphicsPipeline(pub(crate) GraphicsPipeline, pub(crate) vk::Pipeline);

impl ConcreteGraphicsPipeline {
    pub fn get_handle(&self) -> vk::Pipeline {
        self.1
    }
}
