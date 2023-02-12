mod passes;
pub use passes::*;

use pumice::VulkanResult;

use crate::{
    device::Device,
    graph::{compile::GraphContext, execute::GraphExecutor, record::GraphPassBuilder},
};

pub trait RenderPass: 'static + Send {
    fn prepare(&mut self) {}
    unsafe fn execute(&mut self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()>;
}

pub trait CreatePass: 'static + Send {
    type PreparedData: 'static + Send;
    fn prepare(&mut self, builder: &mut GraphPassBuilder) -> Self::PreparedData;
    fn create(
        self,
        prepared: Self::PreparedData,
        ctx: &mut GraphContext,
    ) -> Box<dyn RenderPass + Send>;
}

impl RenderPass for () {
    unsafe fn execute(&mut self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()> {
        VulkanResult::Ok(())
    }
}

impl<P: RenderPass + Send, F: FnMut(&mut GraphPassBuilder) -> P + Send + 'static> CreatePass for F {
    type PreparedData = P;
    fn prepare(&mut self, builder: &mut GraphPassBuilder) -> Self::PreparedData {
        self(builder)
    }
    fn create(
        self,
        prepared: Self::PreparedData,
        ctx: &mut GraphContext,
    ) -> Box<dyn RenderPass + Send> {
        Box::new(prepared)
    }
}

impl<F: FnMut(&GraphExecutor, &Device) + Send + 'static> RenderPass for F {
    unsafe fn execute(&mut self, executor: &GraphExecutor, device: &Device) -> VulkanResult<()> {
        self(executor, device);
        Ok(())
    }
}
