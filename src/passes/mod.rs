mod passes;
pub use passes::*;

use pumice::VulkanResult;

use crate::{
    device::Device,
    graph::{compile::GraphContext, execute::GraphExecutor, record::GraphPassBuilder},
};

pub trait RenderPass: 'static {
    fn prepare(&mut self);
    unsafe fn execute(&mut self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()>;
}

pub trait CreatePass: 'static {
    type PreparedData: 'static;
    fn prepare(&mut self, builder: &mut GraphPassBuilder) -> Self::PreparedData;
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

impl<P: RenderPass, F: FnMut(&mut GraphPassBuilder) -> P + 'static> CreatePass for F {
    type PreparedData = P;
    fn prepare(&mut self, builder: &mut GraphPassBuilder) -> Self::PreparedData {
        self(builder)
    }
    fn create(self, prepared: Self::PreparedData, ctx: &mut GraphContext) -> Box<dyn RenderPass> {
        Box::new(prepared)
    }
}
