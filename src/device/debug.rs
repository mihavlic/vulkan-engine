use std::{ffi::c_char, fmt::Display};

use pumice::{util::ObjectHandle, vk};
use smallvec::SmallVec;

use crate::device::Device;

pub fn with_temporary_cstr<F: FnOnce(*const c_char)>(display: &dyn Display, fun: F) {
    let mut name_buf: SmallVec<[u8; 64]> = SmallVec::new();
    use std::io::Write;
    write!(name_buf, "{display}\0");
    fun(name_buf.as_ptr().cast());
}

pub fn maybe_attach_debug_label<H: ObjectHandle>(handle: H, name: &dyn Display, device: &Device) {
    if device.debug() {
        with_temporary_cstr(name, |cstr| {
            let info = vk::DebugUtilsObjectNameInfoEXT {
                object_type: H::TYPE,
                object_handle: handle.as_raw(),
                p_object_name: cstr,
                ..Default::default()
            };
            unsafe {
                device.device().set_debug_utils_object_name_ext(&info);
            }
        })
    }
}

#[repr(transparent)]
pub struct DisplayConcat<'a>([&'a dyn Display]);

impl<'a> DisplayConcat<'a> {
    pub fn new<'b>(slice: &'b [&'a dyn Display]) -> &'b Self {
        unsafe { std::mem::transmute::<&'b [&'a dyn Display], &'b DisplayConcat<'a>>(slice) }
    }
}

impl<'a> Display for DisplayConcat<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for d in &self.0 {
            d.fmt(f)?;
        }
        Ok(())
    }
}

pub struct LazyDisplay<F: Fn(&mut std::fmt::Formatter<'_>) -> std::fmt::Result>(pub F);

impl<F: Fn(&mut std::fmt::Formatter<'_>) -> std::fmt::Result> Display for LazyDisplay<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (self.0)(f)
    }
}
