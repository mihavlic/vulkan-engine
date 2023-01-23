use pumice::vk;

use crate::{
    graph::{task::GraphicsPipelinePromise, GraphContext, GraphImage},
    object::{ConcreteGraphicsPipeline, GraphicsPipeline, RenderPassMode},
};

use super::{CreatePass, RenderPass};

pub struct ClearImage {
    pub image: GraphImage,
    pub color: vk::ClearColorValue,
}

impl CreatePass for ClearImage {
    type PreparedData = ();
    fn prepare(
        &mut self,
        builder: &mut crate::graph::GraphPassBuilder,
        device: &crate::device::Device,
    ) -> Self::PreparedData {
        builder.use_image(
            self.image,
            vk::ImageUsageFlags::TRANSFER_DST,
            vk::PipelineStageFlags2KHR::TRANSFER,
            vk::AccessFlags2KHR::TRANSFER_WRITE,
            vk::ImageLayout::GENERAL,
            None,
        );
    }
    fn create(self, prepared: Self::PreparedData, ctx: &mut GraphContext) -> Box<dyn RenderPass> {
        Box::new(self)
    }
}

impl RenderPass for ClearImage {
    fn prepare(&mut self) {
        {}
    }
    unsafe fn execute(
        &mut self,
        executor: &super::GraphExecutor,
        device: &crate::device::Device,
    ) -> pumice::VulkanResult<()> {
        let cmd = executor.command_buffer();
        let image = executor.get_image(self.image);
        let range = executor.get_image_subresource_range(self.image);
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
    fn prepare(
        &mut self,
        builder: &mut crate::graph::GraphPassBuilder,
        device: &crate::device::Device,
    ) -> Self::PreparedData {
        for &image in &self.attachments {
            builder.use_image(
                image,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::PipelineStageFlags2KHR::FRAGMENT_SHADER,
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

    fn create(self, prepared: Self::PreparedData, ctx: &mut GraphContext) -> Box<dyn RenderPass> {
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
        executor: &crate::graph::GraphExecutor,
        device: &crate::device::Device,
    ) -> pumice::VulkanResult<()> {
        let d = device.device();
        let cmd = executor.command_buffer();

        let info = vk::RenderPassBeginInfo {
            render_pass: todo!(),
            framebuffer: todo!(),
            render_area: todo!(),
            clear_value_count: todo!(),
            p_clear_values: todo!(),
            ..Default::default()
        };

        // vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

        // Draw calls here

        // vkCmdEndRenderPass(command_buffer);
    }
}
