use std::{
    borrow::Cow,
    ffi::{c_char, CStr},
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
    pub module: vk::ShaderModule,
    pub name: Cow<'static, str>,
    pub specialization_info: Option<SpecializationInfo>,
}

impl PipelineStage {
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

// // Is this a bad idea?
// // Todo use an arena allocator
// pub struct GraphicsPipelineCreateInfoScratch {
//     create_infos: SmallVec<[vk::GraphicsPipelineCreateInfo; 4]>,
//     stages: SmallVec<[vk::PipelineShaderStageCreateInfo; 4]>,
//     vertex_input_states: SmallVec<[vk::PipelineVertexInputStateCreateInfo; 4]>,
//     input_assembly_states: SmallVec<[vk::PipelineInputAssemblyStateCreateInfo; 4]>,
//     tessellation_states: SmallVec<[vk::PipelineTessellationStateCreateInfo; 4]>,
//     viewport_states: SmallVec<[vk::PipelineViewportStateCreateInfo; 4]>,
//     rasterization_states: SmallVec<[vk::PipelineRasterizationStateCreateInfo; 4]>,
//     multisample_states: SmallVec<[vk::PipelineMultisampleStateCreateInfo; 4]>,
//     depth_stencil_states: SmallVec<[vk::PipelineDepthStencilStateCreateInfo; 4]>,
//     color_blend_states: SmallVec<[vk::PipelineColorBlendStateCreateInfo; 4]>,
//     dynamic_states: SmallVec<[vk::PipelineDynamicStateCreateInfo; 4]>,
//     dynamic_rendering_infos: SmallVec<[vk::PipelineRenderingCreateInfoKHR; 4]>,
// }

// impl GraphicsPipelineCreateInfoScratch {
//     fn get_create_infos(&self) -> &[vk::GraphicsPipelineCreateInfo] {
//         &self.create_infos
//     }
// }

struct CStrIterator<'a>(Option<slice::Iter<'a, u8>>);

impl<'a> Iterator for CStrIterator<'a> {
    type Item = c_char;
    fn next(&mut self) -> Option<Self::Item> {
        match self.0 {
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
        match self.0 {
            Some(iter) => iter.len() + 1,
            None => 0,
        }
    }

    // fn is_empty(&self) -> bool {
    //     self.0.is_none()
    // }
}

impl<'a> CStrIterator<'a> {
    pub fn new(bytes: impl AsRef<[u8]>) -> Self {
        Self(Some(bytes.as_ref().into_iter()))
    }
}

struct DropClosure<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> Drop for DropClosure<F> {
    fn drop(&mut self) {
        (self.0.take().unwrap())();
    }
}

impl GraphicsPipelineCreateInfo {
    pub unsafe fn create_multiple<I: IntoIterator<Item = Self>>(
        this: I,
        pipeline_cache: vk::PipelineCache,
        bump: &mut Bump,
        ctx: &Device,
    ) -> VulkanResult<(Vec<vk::Pipeline>, vk::Result)>
    where
        I::IntoIter: ExactSizeIterator,
    {
        let this = this.into_iter();
        assert!(this.len() > 0);

        let mut delayed_drop: SmallVec<[(*const (), std::alloc::Layout); 4]> = SmallVec::new();

        fn delay_dealloc<T>(
            vec: Vec<T>,
            delay: &mut SmallVec<[(*const (), std::alloc::Layout); 4]>,
        ) -> (*const T, u32) {
            if vec.is_empty() {
                return (std::ptr::null(), 0);
            }

            let (ptr, len, capacity) = (vec.as_ptr(), vec.len(), vec.capacity());
            std::mem::forget(vec);

            let layout = std::alloc::Layout::array::<T>(capacity).unwrap();
            delay.push((ptr.cast(), layout));

            (ptr, len as u32)
        }

        fn delay_dealloc_small<T, A: Array<Item = T>>(
            vec: SmallVec<A>,
            delay: &mut SmallVec<[(*const (), std::alloc::Layout); 4]>,
            bump: &Bump,
        ) -> (*const T, u32) {
            if vec.is_empty() {
                return (std::ptr::null(), 0);
            }

            if vec.spilled() {
                let (ptr, len, capacity) = (vec.as_ptr(), vec.len(), vec.capacity());
                std::mem::forget(vec);

                let layout = std::alloc::Layout::array::<T>(capacity).unwrap();
                delay.push((ptr.cast(), layout));

                (ptr, len as u32)
            } else {
                let len = vec.len();
                let ptr = bump.alloc_slice_fill_iter(vec.into_iter());
                (ptr.as_ptr(), len as u32)
            }
        }

        let create_infos = this.map(|pipeline| {
            let mut pnext_head: *const std::ffi::c_void = std::ptr::null();

            macro_rules! add_pnext {
                ($head:expr) => {
                    let ptr = &mut $head;
                    ptr.p_next = pnext_head;
                    pnext_head = ptr as *const _ as *const std::ffi::c_void;
                };
            }

            let mut handle = None;
            let (render_pass, subpass) = match pipeline.render_pass {
                RenderPassMode::Normal {
                    render_pass,
                    subpass,
                } => {
                    handle = Some(render_pass);
                    (handle.as_ref().unwrap().0.get_handle(), subpass)
                }
                RenderPassMode::Dynamic {
                    view_mask,
                    colors,
                    depth,
                    stencil,
                } => {
                    let (p_colors, colors_len) =
                        delay_dealloc_small(colors, &mut delayed_drop, bump);
                    let info = vk::PipelineRenderingCreateInfoKHR {
                        view_mask,
                        color_attachment_count: colors_len,
                        p_color_attachment_formats: p_colors,
                        depth_attachment_format: depth,
                        stencil_attachment_format: stencil,
                        ..Default::default()
                    };
                    let info = bump.alloc(info);
                    add_pnext!(info);
                    (vk::RenderPass::null(), 0)
                }
            };

            let stages = bump.alloc_slice_fill_iter(pipeline.stages.into_iter().map(|s| {
                const MAIN_CSTR: &CStr = pumice::cstr!("main");
                let name = match s.name.as_ref() {
                    "main" => MAIN_CSTR.as_ptr(),
                    other => bump
                        .alloc_slice_fill_iter(CStrIterator::new(s.name.as_bytes()))
                        .as_ptr() as *const c_char,
                };
                let info = |i: SpecializationInfo| {
                    let data_len = i.data.len();
                    let (p_entries, entries_len) = delay_dealloc(i.map_entries, &mut delayed_drop);
                    let (p_data, _) = delay_dealloc(i.data, &mut delayed_drop);
                    let info = vk::SpecializationInfo {
                        map_entry_count: entries_len,
                        p_map_entries: p_entries,
                        data_size: data_len,
                        p_data: p_data.cast(),
                    };
                    bump.alloc(info) as *const _
                };
                let p_specialization_info =
                    s.specialization_info.map(info).unwrap_or(std::ptr::null());
                vk::PipelineShaderStageCreateInfo {
                    flags: s.flags,
                    stage: s.stage,
                    module: s.module,
                    p_name: name,
                    p_specialization_info,
                    ..Default::default()
                }
            }));

            let vertex_input = {
                let a = pipeline.vertex_input;
                let (p_vertex_binding_descriptions, vertex_binding_description_count) =
                    delay_dealloc(a.vertex_bindings, &mut delayed_drop);
                let (p_vertex_attribute_descriptions, vertex_attribute_description_count) =
                    delay_dealloc(a.vertex_attributes, &mut delayed_drop);
                let info = vk::PipelineVertexInputStateCreateInfo {
                    vertex_binding_description_count,
                    p_vertex_binding_descriptions,
                    vertex_attribute_description_count,
                    p_vertex_attribute_descriptions,
                    ..Default::default()
                };
                bump.alloc(info)
            };

            let input_assembly = {
                let a = pipeline.input_assembly;
                let info = vk::PipelineInputAssemblyStateCreateInfo {
                    topology: a.topology,
                    primitive_restart_enable: a.primitive_restart_enable as vk::Bool32,
                    ..Default::default()
                };
                bump.alloc(info)
            };

            let tessellation = {
                let a = pipeline.tessellation;
                let info = vk::PipelineTessellationStateCreateInfo {
                    patch_control_points: a.patch_control_points,
                    ..Default::default()
                };
                bump.alloc(info)
            };

            let viewport = {
                let a = pipeline.viewport;
                let (p_viewports, viewport_count) = delay_dealloc(a.viewports, &mut delayed_drop);
                let (p_scissors, scissor_count) = delay_dealloc(a.scissors, &mut delayed_drop);
                let info = vk::PipelineViewportStateCreateInfo {
                    viewport_count,
                    p_viewports,
                    scissor_count,
                    p_scissors,
                    ..Default::default()
                };
                bump.alloc(info)
            };

            let rasterization = {
                let a = pipeline.rasterization;
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
                let a = pipeline.multisample;
                let p_sample_mask = match a.sample_mask {
                    Some(s) => {
                        let (ptr, _) = delay_dealloc(s, &mut delayed_drop);
                        ptr
                    }
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
                let a = pipeline.depth_stencil;
                let info = vk::PipelineDepthStencilStateCreateInfo {
                    depth_test_enable: a.depth_test_enable as vk::Bool32,
                    depth_write_enable: a.depth_write_enable as vk::Bool32,
                    depth_compare_op: a.depth_compare_op,
                    depth_bounds_test_enable: a.depth_bounds_test_enable as vk::Bool32,
                    stencil_test_enable: a.stencil_test_enable as vk::Bool32,
                    front: a.front,
                    back: a.back,
                    min_depth_bounds: a.min_depth_bounds,
                    max_depth_bounds: a.max_depth_bounds,
                    ..Default::default()
                };
                bump.alloc(info)
            };

            let color_blend = {
                let a = pipeline.color_blend;
                let (p_attachments, attachment_count) =
                    delay_dealloc(a.attachments, &mut delayed_drop);
                let info = vk::PipelineColorBlendStateCreateInfo {
                    logic_op_enable: a.logic_op_enable as vk::Bool32,
                    logic_op: a.logic_op,
                    attachment_count,
                    p_attachments,
                    blend_constants: a.blend_constants,
                    ..Default::default()
                };
                bump.alloc(info)
            };

            let dynamic_state = {
                let a = pipeline.dynamic_state;
                match a {
                    None => std::ptr::null(),
                    Some(a) => {
                        let (p_dynamic_states, dynamic_state_count) =
                            delay_dealloc(a.dynamic_states, &mut delayed_drop);
                        let info = vk::PipelineDynamicStateCreateInfo {
                            dynamic_state_count,
                            p_dynamic_states,
                            ..Default::default()
                        };
                        bump.alloc(info)
                    }
                }
            };

            let (base_pipeline_handle, base_pipeline_index) = match &pipeline.base_pipeline {
                BasePipeline::Index(index) => (vk::Pipeline::null(), *index),
                BasePipeline::Pipeline(handle) => (handle.get_handle(), -1),
                BasePipeline::None => (vk::Pipeline::null(), -1),
            };

            vk::GraphicsPipelineCreateInfo {
                p_next: pnext_head,
                flags: pipeline.flags,
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
                layout: pipeline.layout.0.get_handle(),
                render_pass,
                subpass,
                base_pipeline_handle,
                base_pipeline_index,
                ..Default::default()
            }
        });

        let create_infos = bump.alloc_slice_fill_iter(create_infos);

        ctx.device().create_graphics_pipelines(
            pipeline_cache,
            create_infos,
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

    type InputData<'a> = GraphicsPipelineCreateInfo;
    type Data = GraphicsPipelineState;

    unsafe fn create<'a>(
        data: Self::InputData<'a>,
        ctx: &Self::Parent,
    ) -> VulkanResult<Self::Data> {
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
