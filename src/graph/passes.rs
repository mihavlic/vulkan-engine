use pumice::vk;

use super::{CreatePass, GraphImage, RenderPass};

pub struct ClearImage {
    pub image: GraphImage,
    pub color: vk::ClearColorValue,
}

impl CreatePass for ClearImage {
    type Pass = Self;
    fn create(
        self,
        builder: &mut super::GraphPassBuilder,
        device: &crate::device::Device,
    ) -> Self::Pass {
        builder.use_image(
            self.image,
            vk::ImageUsageFlags::TRANSFER_DST,
            vk::PipelineStageFlags2KHR::TRANSFER,
            vk::AccessFlags2KHR::TRANSFER_WRITE,
            vk::ImageLayout::GENERAL,
            None,
        );
        self
    }
}

impl RenderPass for ClearImage {
    fn prepare(&mut self) {
        {}
    }
    unsafe fn execute(
        self,
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
