/// A utility trait to make possibly empty containers into pointers for ffi.
/// This is really needed because option lacks a `as_ptr()` function
/// and an empty slice returns a dangling pointer, this should be fine because it
/// at the same time has a `len` of 0 but it's just bad practice to rely only on the length.
pub trait AsFFiPtr {
    type Pointee;
    fn as_ffi_ptr(&self) -> *const Self::Pointee;
    fn as_ffi_ptr_mut(&mut self) -> *mut Self::Pointee;
}

impl<T> AsFFiPtr for Option<T> {
    type Pointee = T;
    #[inline(always)]
    fn as_ffi_ptr(&self) -> *const Self::Pointee {
        match self {
            Some(s) => s,
            None => std::ptr::null(),
        }
    }
    #[inline(always)]
    fn as_ffi_ptr_mut(&mut self) -> *mut Self::Pointee {
        match self {
            Some(s) => s,
            None => std::ptr::null_mut(),
        }
    }
}

impl<T> AsFFiPtr for [T] {
    type Pointee = T;
    #[inline(always)]
    fn as_ffi_ptr(&self) -> *const Self::Pointee {
        if self.is_empty() {
            std::ptr::null()
        } else {
            self.as_ptr()
        }
    }
    #[inline(always)]
    fn as_ffi_ptr_mut(&mut self) -> *mut Self::Pointee {
        if self.is_empty() {
            std::ptr::null_mut()
        } else {
            self.as_mut_ptr()
        }
    }
}
