use std::{
    borrow::Cow,
    ffi::{c_char, c_void, CStr},
    hash::{Hash, Hasher},
    ptr, slice,
    sync::Arc,
};

use crate::{
    arena::uint::OptionalU32,
    device::{batch::GenerationId, Device},
    storage::{constant_ahash_hasher, nostore::SimpleStorage, MutableShared, SynchronizationLock},
    util::ffi_ptr::AsFFiPtr,
};
use bumpalo::Bump;
use parking_lot::{RawMutex, RawRwLock};
use pumice::{util::ObjectHandle, vk, VulkanResult};
use smallvec::{Array, SmallVec};

use super::{ArcHandle, BasicObjectData, Object, ObjectData};

#[derive(Clone)]
pub struct SpecializationInfo {
    pub map_entries: Vec<vk::SpecializationMapEntry>,
    pub data: Vec<u8>,
}

#[derive(Clone)]
pub struct PipelineStage {
    pub flags: vk::PipelineShaderStageCreateFlags,
    pub stage: vk::ShaderStageFlags,
    pub module: super::ShaderModule,
    pub name: Cow<'static, str>,
    pub specialization_info: Option<SpecializationInfo>,
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
    Delayed,
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
            RenderPassMode::Delayed => {
                panic!("RenderPassMode::Delayed should only be used for Pipeline handle creation")
            }
        }
    }
}

#[derive(Clone, Default)]
pub enum BasePipeline {
    Index(i32),
    Pipeline(ConcreteGraphicsPipeline),
    #[default]
    None,
}

pub mod state {
    use pumice::vk;

    pub type InputBinding = vk::VertexInputBindingDescription;
    pub type InputAttribute = vk::VertexInputAttributeDescription;

    #[derive(Clone, Default)]
    pub struct VertexInput {
        pub vertex_bindings: Vec<InputBinding>,
        pub vertex_attributes: Vec<InputAttribute>,
    }

    #[derive(Clone, Default)]
    pub struct InputAssembly {
        pub topology: vk::PrimitiveTopology,
        pub primitive_restart_enable: bool,
    }

    #[derive(Clone, Default)]
    pub struct Tessellation {
        pub patch_control_points: u32,
    }

    #[derive(Clone, Default)]
    pub struct Viewport {
        pub viewports: Vec<vk::Viewport>,
        pub scissors: Vec<vk::Rect2D>,
    }

    #[derive(Clone, Default)]
    pub struct Rasterization {
        pub depth_clamp_enable: bool,
        pub rasterizer_discard_enable: bool,
        pub polygon_mode: vk::PolygonMode,
        pub cull_mode: vk::CullModeFlags,
        pub front_face: vk::FrontFace,
        pub depth_bias_enable: bool,
        pub depth_bias_constant_factor: f32,
        pub depth_bias_clamp: f32,
        pub depth_bias_slope_factor: f32,
        pub line_width: f32,
    }

    #[derive(Clone, Default)]
    pub struct Multisample {
        pub rasterization_samples: vk::SampleCountFlags,
        pub sample_shading_enable: bool,
        pub min_sample_shading: f32,
        pub sample_mask: Option<Vec<vk::SampleMask>>,
        pub alpha_to_coverage_enable: bool,
        pub alpha_to_one_enable: bool,
    }

    #[derive(Clone, Default)]
    pub struct DepthStencil {
        pub depth_test_enable: bool,
        pub depth_write_enable: bool,
        pub depth_compare_op: vk::CompareOp,
        pub depth_bounds_test_enable: bool,
        pub stencil_test_enable: bool,
        pub front: vk::StencilOpState,
        pub back: vk::StencilOpState,
        pub min_depth_bounds: f32,
        pub max_depth_bounds: f32,
    }

    pub type Attachment = vk::PipelineColorBlendAttachmentState;

    #[derive(Clone, Default)]
    pub struct ColorBlend {
        pub logic_op_enable: bool,
        pub logic_op: vk::LogicOp,
        pub attachments: Vec<Attachment>,
        pub blend_constants: [f32; 4],
    }

    #[derive(Clone, Default)]
    pub struct DynamicState {
        pub dynamic_states: Vec<vk::DynamicState>,
    }
}

#[derive(Clone)]
pub struct GraphicsPipelineCreateInfo {
    pub flags: vk::PipelineCreateFlags,
    pub stages: Vec<PipelineStage>,

    pub vertex_input: state::VertexInput,
    pub input_assembly: state::InputAssembly,
    pub tessellation: state::Tessellation,
    pub viewport: state::Viewport,
    pub rasterization: state::Rasterization,
    pub multisample: state::Multisample,
    pub depth_stencil: state::DepthStencil,
    pub color_blend: state::ColorBlend,
    pub dynamic_state: Option<state::DynamicState>,

    pub layout: super::PipelineLayout,
    pub render_pass: RenderPassMode,
    pub base_pipeline: BasePipeline,
}

struct CStrIterator<'a>(Option<slice::Iter<'a, u8>>);

impl<'a> Iterator for CStrIterator<'a> {
    type Item = c_char;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            Some(iter) => match iter.next() {
                Some(next) => Some(unsafe { std::mem::transmute::<u8, c_char>(*next) }),
                None => {
                    self.0 = None;
                    Some(0)
                }
            },
            None => None,
        }
    }
}

impl<'a> ExactSizeIterator for CStrIterator<'a> {
    fn len(&self) -> usize {
        match &self.0 {
            Some(iter) => iter.len() + 1,
            None => 0,
        }
    }

    // fn is_empty(&self) -> bool {
    //     self.0.is_none()
    // }
}

impl<'a> CStrIterator<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self(Some(bytes.into_iter()))
    }
}

struct DropClosure<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> Drop for DropClosure<F> {
    fn drop(&mut self) {
        (self.0.take().unwrap())();
    }
}

macro_rules! add_pnext {
    ($head:expr, $next:expr) => {
        $next.p_next = $head;
        $head = $next as *const _ as *const std::ffi::c_void;
    };
}

impl GraphicsPipelineCreateInfo {
    pub unsafe fn to_vk(&self, bump: &Bump, ctx: &Device) -> vk::GraphicsPipelineCreateInfo {
        let mut pnext_head: *const std::ffi::c_void = std::ptr::null();

        let (render_pass, subpass) =
            raw_info_handle_renderpass(&self.render_pass, &mut pnext_head, bump);

        let stages = bump.alloc_slice_fill_iter(self.stages.iter().map(|s| {
            const MAIN_CSTR: &CStr = pumice::cstr!("main");
            let name = match s.name.as_ref() {
                "main" => MAIN_CSTR.as_ptr(),
                other => bump
                    .alloc_slice_fill_iter(CStrIterator::new(s.name.as_bytes()))
                    .as_ptr() as *const c_char,
            };
            let info = |i: &SpecializationInfo| {
                let info = vk::SpecializationInfo {
                    map_entry_count: i.map_entries.len() as u32,
                    p_map_entries: i.map_entries.as_ffi_ptr(),
                    data_size: i.data.len(),
                    p_data: i.data.as_ffi_ptr().cast(),
                };
                bump.alloc(info) as *const _
            };
            let p_specialization_info = s
                .specialization_info
                .as_ref()
                .map(info)
                .unwrap_or(std::ptr::null());
            vk::PipelineShaderStageCreateInfo {
                flags: s.flags,
                stage: s.stage,
                module: s.module.raw(),
                p_name: name,
                p_specialization_info,
                ..Default::default()
            }
        }));

        let vertex_input = {
            let a = &self.vertex_input;
            let info = vk::PipelineVertexInputStateCreateInfo {
                vertex_binding_description_count: a.vertex_bindings.len() as u32,
                p_vertex_binding_descriptions: a.vertex_bindings.as_ffi_ptr(),
                vertex_attribute_description_count: a.vertex_attributes.len() as u32,
                p_vertex_attribute_descriptions: a.vertex_attributes.as_ffi_ptr(),
                ..Default::default()
            };
            bump.alloc(info)
        };

        let input_assembly = {
            let a = &self.input_assembly;
            let info = vk::PipelineInputAssemblyStateCreateInfo {
                topology: a.topology,
                primitive_restart_enable: a.primitive_restart_enable as vk::Bool32,
                ..Default::default()
            };
            bump.alloc(info)
        };

        let tessellation = {
            let a = &self.tessellation;
            let info = vk::PipelineTessellationStateCreateInfo {
                patch_control_points: a.patch_control_points,
                ..Default::default()
            };
            bump.alloc(info)
        };

        let viewport = {
            let a = &self.viewport;
            let info = vk::PipelineViewportStateCreateInfo {
                viewport_count: a.viewports.len() as u32,
                p_viewports: a.viewports.as_ffi_ptr(),
                scissor_count: a.scissors.len() as u32,
                p_scissors: a.scissors.as_ffi_ptr(),
                ..Default::default()
            };
            bump.alloc(info)
        };

        let rasterization = {
            let a = &self.rasterization;
            let info = vk::PipelineRasterizationStateCreateInfo {
                depth_clamp_enable: a.depth_clamp_enable as vk::Bool32,
                rasterizer_discard_enable: a.rasterizer_discard_enable as vk::Bool32,
                polygon_mode: a.polygon_mode,
                cull_mode: a.cull_mode,
                front_face: a.front_face,
                depth_bias_enable: a.depth_bias_enable as vk::Bool32,
                depth_bias_constant_factor: a.depth_bias_constant_factor,
                depth_bias_clamp: a.depth_bias_clamp,
                depth_bias_slope_factor: a.depth_bias_slope_factor,
                line_width: a.line_width,
                ..Default::default()
            };
            bump.alloc(info)
        };

        let multisample = {
            let a = &self.multisample;
            let p_sample_mask = match &a.sample_mask {
                Some(s) => s.as_ffi_ptr(),
                None => std::ptr::null(),
            };
            let info = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: a.rasterization_samples,
                sample_shading_enable: a.sample_shading_enable as vk::Bool32,
                min_sample_shading: a.min_sample_shading,
                p_sample_mask,
                alpha_to_coverage_enable: a.alpha_to_coverage_enable as vk::Bool32,
                alpha_to_one_enable: a.alpha_to_one_enable as vk::Bool32,
                ..Default::default()
            };
            bump.alloc(info)
        };

        let depth_stencil = {
            let a = &self.depth_stencil;
            let info = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: a.depth_test_enable as vk::Bool32,
                depth_write_enable: a.depth_write_enable as vk::Bool32,
                depth_compare_op: a.depth_compare_op,
                depth_bounds_test_enable: a.depth_bounds_test_enable as vk::Bool32,
                stencil_test_enable: a.stencil_test_enable as vk::Bool32,
                front: a.front.clone(),
                back: a.back.clone(),
                min_depth_bounds: a.min_depth_bounds,
                max_depth_bounds: a.max_depth_bounds,
                ..Default::default()
            };
            bump.alloc(info)
        };

        let color_blend = {
            let a = &self.color_blend;
            let info = vk::PipelineColorBlendStateCreateInfo {
                logic_op_enable: a.logic_op_enable as vk::Bool32,
                logic_op: a.logic_op,
                attachment_count: a.attachments.len() as u32,
                p_attachments: a.attachments.as_ffi_ptr(),
                blend_constants: a.blend_constants,
                ..Default::default()
            };
            bump.alloc(info)
        };

        let dynamic_state = {
            let a = &self.dynamic_state;
            match a {
                None => std::ptr::null(),
                Some(a) => {
                    let info = vk::PipelineDynamicStateCreateInfo {
                        dynamic_state_count: a.dynamic_states.len() as u32,
                        p_dynamic_states: a.dynamic_states.as_ffi_ptr(),
                        ..Default::default()
                    };
                    bump.alloc(info)
                }
            }
        };

        let (base_pipeline_handle, base_pipeline_index) = match &self.base_pipeline {
            BasePipeline::Index(index) => (vk::Pipeline::null(), *index),
            BasePipeline::Pipeline(handle) => (handle.get_handle(), -1),
            BasePipeline::None => (vk::Pipeline::null(), -1),
        };

        vk::GraphicsPipelineCreateInfo {
            p_next: pnext_head,
            flags: self.flags,
            stage_count: stages.len() as u32,
            p_stages: stages.as_ffi_ptr(),
            p_vertex_input_state: vertex_input,
            p_input_assembly_state: input_assembly,
            p_tessellation_state: tessellation,
            p_viewport_state: viewport,
            p_rasterization_state: rasterization,
            p_multisample_state: multisample,
            p_depth_stencil_state: depth_stencil,
            p_color_blend_state: color_blend,
            p_dynamic_state: dynamic_state,
            layout: self.layout.0.get_handle(),
            render_pass,
            subpass,
            base_pipeline_handle,
            base_pipeline_index,
            ..Default::default()
        }
    }
}

pub(crate) fn raw_info_handle_renderpass(
    mode: &RenderPassMode,
    pnext: &mut *const c_void,
    bump: &Bump,
) -> (vk::RenderPass, u32) {
    match *mode {
        RenderPassMode::Normal {
            ref render_pass,
            subpass,
        } => (render_pass.raw(), subpass),
        RenderPassMode::Dynamic {
            view_mask,
            ref colors,
            depth,
            stencil,
        } => {
            let info = vk::PipelineRenderingCreateInfoKHR {
                s_type: pumice::vk10::StructureType::PIPELINE_RENDERING_CREATE_INFO_KHR,
                view_mask,
                color_attachment_count: colors.len() as u32,
                p_color_attachment_formats: colors.as_ffi_ptr(),
                depth_attachment_format: depth,
                stencil_attachment_format: stencil,
                ..Default::default()
            };
            let info = bump.alloc(info);

            add_pnext!((*pnext), info);

            (vk::RenderPass::null(), 0)
        }
        // the user must fill these themselves
        RenderPassMode::Delayed => (vk::RenderPass::null(), 0),
    }
}

enum GraphicsPipelineEntryHandle {
    Promised(Arc<RawRwLock>),
    Created(vk::Pipeline, RenderPassMode),
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
                GraphicsPipelineEntryHandle::Created(ok, _) => GetPipelineResult::Ready(*ok),
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
    pub(crate) unsafe fn add_promised_pipeline(
        &mut self,
        handle: vk::Pipeline,
        mode_hash: u64,
        mode: RenderPassMode,
    ) {
        if let Some(found) = self.pipelines.iter_mut().find(|e| e.mode_hash == mode_hash) {
            match &found.handle {
                GraphicsPipelineEntryHandle::Promised(mutex) => {
                    found.handle = GraphicsPipelineEntryHandle::Created(handle, mode);
                }
                GraphicsPipelineEntryHandle::Created(_, _) => {
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
                GraphicsPipelineEntryHandle::Created(handle, _) => {
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

    type InputData<'a> = GraphicsPipelineCreateInfo;
    type Data = GraphicsPipelineState;

    unsafe fn create<'a>(
        data: Self::InputData<'a>,
        ctx: &Self::Parent,
    ) -> VulkanResult<Self::Data> {
        // TODO allow fully concrete pipelines?
        assert!(
            matches!(data.render_pass, RenderPassMode::Delayed),
            "This code path currently only supports delayed pipeline creation"
        );
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
        &parent.graphics_pipelines
    }
}

#[derive(Clone)]
pub struct ConcreteGraphicsPipeline(pub(crate) GraphicsPipeline, pub(crate) vk::Pipeline);

impl ConcreteGraphicsPipeline {
    pub fn get_handle(&self) -> vk::Pipeline {
        self.1
    }
}
