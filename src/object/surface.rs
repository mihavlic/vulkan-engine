use pumice::vk;

use crate::context::instance::Instance;

pub struct Surface {
    handle: vk::SurfaceKHR,
    instance: Instance,
}

impl Surface {
    pub fn from_raw(handle: vk::SurfaceKHR, instance: Instance) -> Self {
        Self { handle, instance }
    }
    pub fn handle(&self) -> vk::SurfaceKHR {
        self.handle
    }
    pub unsafe fn to_raw(self) -> vk::SurfaceKHR {
        let handle = self.handle;
        std::mem::forget(self);
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
