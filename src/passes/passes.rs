use pumice::{util::ObjectHandle, vk};
use smallvec::{smallvec, SmallVec};

use crate::{
    device::Device,
    graph::{
        compile::GraphContext, execute::GraphExecutor, record::GraphPassBuilder,
        task::GraphicsPipelinePromise, GraphImage,
    },
    object::{self, ConcreteGraphicsPipeline, Extent, GraphicsPipeline, RenderPassMode},
};

use super::{CreatePass, RenderPass};

pub struct ClearImage {
    pub image: GraphImage,
    pub color: vk::ClearColorValue,
}

impl CreatePass for ClearImage {
    type PreparedData = ();
    fn prepare(&mut self, builder: &mut GraphPassBuilder) -> Self::PreparedData {
        builder.use_image(
            self.image,
            vk::ImageUsageFlags::TRANSFER_DST,
            vk::PipelineStageFlags2KHR::TRANSFER,
            vk::AccessFlags2KHR::TRANSFER_WRITE,
            vk::ImageLayout::GENERAL,
            None,
        );
    }
    fn create(
        self,
        prepared: Self::PreparedData,
        ctx: &mut GraphContext,
    ) -> Box<dyn RenderPass + Send> {
        Box::new(self)
    }
}

impl RenderPass for ClearImage {
    fn prepare(&mut self) {
        {}
    }
    unsafe fn execute(
        &mut self,
        executor: &GraphExecutor,
        device: &Device,
    ) -> pumice::VulkanResult<()> {
        let cmd = executor.command_buffer();
        let image = executor.get_image(self.image);
        let range = executor.get_image_subresource_range(self.image, vk::ImageAspectFlags::COLOR);
        device.device().cmd_clear_color_image(
            cmd,
            image,
            vk::ImageLayout::GENERAL,
            &self.color,
            std::slice::from_ref(&range),
        );
        Ok(())
    }
}

pub struct SimpleShader {
    pub pipeline: GraphicsPipeline,
    pub attachments: Vec<GraphImage>,
}

impl CreatePass for SimpleShader {
    type PreparedData = GraphicsPipelinePromise;
    fn prepare(&mut self, builder: &mut GraphPassBuilder) -> Self::PreparedData {
        for &image in &self.attachments {
            builder.use_image(
                image,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                None,
            );
        }
        let mode = RenderPassMode::Dynamic {
            view_mask: 0,
            colors: self
                .attachments
                .iter()
                .map(|i| builder.get_image_format(*i))
                .collect(),
            depth: vk::Format::UNDEFINED,
            stencil: vk::Format::UNDEFINED,
        };

        builder.compile_graphics_pipeline(&self.pipeline, &mode)
    }

    fn create(
        self,
        prepared: Self::PreparedData,
        ctx: &mut GraphContext,
    ) -> Box<dyn RenderPass + Send> {
        let pipeline = ctx.resolve_graphics_pipeline(prepared);
        Box::new(SimpleShaderPass {
            info: self,
            pipeline,
        })
    }
}

struct SimpleShaderPass {
    info: SimpleShader,
    pipeline: ConcreteGraphicsPipeline,
}

impl RenderPass for SimpleShaderPass {
    fn prepare(&mut self) {
        {}
    }
    unsafe fn execute(
        &mut self,
        executor: &GraphExecutor,
        device: &Device,
    ) -> pumice::VulkanResult<()> {
        let d = device.device();
        let cmd = executor.command_buffer();

        let attachments = self
            .info
            .attachments
            .iter()
            .map(|&image| vk::RenderingAttachmentInfoKHR {
                image_view: executor.get_default_image_view(image, vk::ImageAspectFlags::COLOR),
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                resolve_mode: vk::ResolveModeFlagsKHR::NONE,
                load_op: vk::AttachmentLoadOp::LOAD,
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value: vk::ClearValue {
                    color: vk::ClearColorValue {
                        float_32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                ..Default::default()
            })
            .collect::<SmallVec<[_; 8]>>();

        let (width, height) = match executor.get_image_extent(self.info.attachments[0]) {
            Extent::D2(w, h) => (w, h),
            Extent::D1(_) | Extent::D3(_, _, _) => panic!("Attachments must be 2D images"),
        };

        let info = vk::RenderingInfoKHR {
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            },
            layer_count: 1,
            view_mask: 0,
            color_attachment_count: attachments.len() as u32,
            p_color_attachments: attachments.as_ptr(),
            p_depth_attachment: std::ptr::null(),
            p_stencil_attachment: std::ptr::null(),
            ..Default::default()
        };

        d.cmd_begin_rendering_khr(cmd, &info);

        d.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.get_handle(),
        );

        // TODO this kinda sucks, we should probably include a bitmask of dynamic states or something
        let pipeline_info = self.pipeline.get_object().0.get_create_info();

        if let Some(state) = &pipeline_info.dynamic_state {
            if state.dynamic_states.contains(&vk::DynamicState::VIEWPORT) {
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: width as f32,
                    height: height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                d.cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport));
            }

            if state.dynamic_states.contains(&vk::DynamicState::SCISSOR) {
                let rect = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width, height },
                };
                d.cmd_set_scissor(cmd, 0, std::slice::from_ref(&rect));
            }
        }

        d.cmd_draw(cmd, 3, 1, 0, 0);

        d.cmd_end_rendering_khr(cmd);

        Ok(())
    }
}
