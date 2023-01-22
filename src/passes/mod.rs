mod passes;
pub use passes::*;

use pumice::VulkanResult;

use crate::{
    device::Device,
    graph::{GraphContext, GraphExecutor, GraphPassBuilder},
};

pub trait RenderPass: 'static {
    fn prepare(&mut self);
    unsafe fn execute(&mut self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()>;
}

pub trait CreatePass: 'static {
    type PreparedData: 'static;
    fn prepare(&mut self, builder: &mut GraphPassBuilder, device: &Device) -> Self::PreparedData;
    fn create(self, prepared: Self::PreparedData, ctx: &mut GraphContext) -> Box<dyn RenderPass>;
}

impl RenderPass for () {
    fn prepare(&mut self) {
        {}
    }
    unsafe fn execute(&mut self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()> {
        VulkanResult::Ok(())
    }
}

impl<P: RenderPass, F: FnMut(&mut GraphPassBuilder, &Device) -> P + 'static> CreatePass for F {
    type PreparedData = P;
    fn prepare(&mut self, builder: &mut GraphPassBuilder, device: &Device) -> Self::PreparedData {
        self(builder, device)
    }
    fn create(self, prepared: Self::PreparedData, ctx: &mut GraphContext) -> Box<dyn RenderPass> {
        Box::new(prepared)
    }
}
