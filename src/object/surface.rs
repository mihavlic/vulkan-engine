use std::mem::ManuallyDrop;

use pumice::vk;

use crate::instance::Instance;

pub struct Surface {
    handle: vk::SurfaceKHR,
    instance: Instance,
}

impl Surface {
    pub fn from_raw(handle: vk::SurfaceKHR, instance: Instance) -> Self {
        Self { handle, instance }
    }
    pub unsafe fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }
    pub unsafe fn into_raw(self) -> vk::SurfaceKHR {
        let prison = ManuallyDrop::new(self);
        let handle = std::ptr::read(&prison.handle);
        let _instance = std::ptr::read(&prison.instance);
        handle
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .handle()
                .destroy_surface_khr(self.handle, self.instance.allocator_callbacks());
        }
    }
}
