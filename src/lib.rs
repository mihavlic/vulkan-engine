#![allow(unused)]

pub mod context;
pub mod object;
pub mod storage;
pub mod synchronization;
pub mod tracing;
pub mod util;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OptionalU32(u32);

impl OptionalU32 {
    pub const NONE: Self = Self(u32::MAX);

    pub const fn new(value: u32) -> Self {
        assert!(value != u32::MAX);
        Self(value)
    }
    pub const fn new_none() -> Self {
        Self::NONE
    }
    pub const fn get(&self) -> Option<u32> {
        if self.0 == u32::MAX {
            None
        } else {
            Some(self.0)
        }
    }
    pub fn set(&mut self, value: Option<u32>) {
        if let Some(value) = value {
            assert!(value != u32::MAX);
            *self = Self(value);
        } else {
            *self = Self::NONE;
        }
    }
}
