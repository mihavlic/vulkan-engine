mod queue;
mod sparse;

pub use {queue::*, sparse::*};

use crate::{
    graph::{self, allocator::round_up_pow2_usize},
    simple_handle,
};
use pumice::vk;
use std::{collections::VecDeque, marker::PhantomData, ptr::NonNull};

use super::{maybe_attach_debug_label, Device};

#[derive(Clone, PartialEq, Eq)]
pub struct SuballocatedMemory {
    pub buffer_offset: usize,
    pub buffer: vk::Buffer,
    pub memory: NonNull<u8>,
}

#[derive(Clone)]
pub struct BufferEntry {
    pub buffer: vk::Buffer,
    pub allocation: pumice_vma::Allocation,
    pub start: NonNull<u8>,
    pub cursor: NonNull<u8>,
    pub end: NonNull<u8>,
}

impl BufferEntry {
    pub unsafe fn bump(
        &mut self,
        layout: std::alloc::Layout,
        device: &Device,
    ) -> Option<(NonNull<u8>, usize)> {
        assert!(self.cursor <= self.end);

        // TODO this is sketchy
        let start_address = self.start.as_ptr() as usize;
        let cursor_address = self.cursor.as_ptr() as usize;
        let aligned_cursor_address = round_up_pow2_usize(cursor_address, layout.align());
        let mut offset = aligned_cursor_address - start_address;

        let offset_align = device
            .physical_device_properties
            .limits
            .min_uniform_buffer_offset_alignment;

        if offset_align > 1 {
            offset = round_up_pow2_usize(offset, offset_align as usize);
        }

        let start_ptr = self.cursor.as_ptr().add(offset);
        let end_ptr = start_ptr.add(layout.size());

        if end_ptr > self.end.as_ptr() {
            return None;
        }

        self.cursor = NonNull::new(end_ptr).unwrap();

        Some((NonNull::new(start_ptr).unwrap(), offset))
    }
}

pub struct RingConfig {
    pub buffer_size: u64,
    pub usage: vk::BufferUsageFlags,
    pub allocation_flags: pumice_vma::AllocationCreateFlags,
    pub required_flags: vk::MemoryPropertyFlags,
    pub preferred_flags: vk::MemoryPropertyFlags,
    pub label: &'static str,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferMonotonic(u64);
impl BufferMonotonic {
    pub fn raw(self) -> u64 {
        self.0
    }
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
}

unsafe fn make_buffer(
    config: &RingConfig,
    device: &Device,
) -> (
    pumice::vk10::Buffer,
    pumice_vma::Allocation,
    NonNull<u8>,
    NonNull<u8>,
) {
    let buffer_info = vk::BufferCreateInfo {
        flags: vk::BufferCreateFlags::empty(),
        size: config.buffer_size,
        usage: config.usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };

    let allocation_info = pumice_vma::AllocationCreateInfo {
        flags: config.allocation_flags,
        usage: pumice_vma::MemoryUsage::Auto,

        ..Default::default()
    };

    let (buffer, allocation, info) = device
        .allocator()
        .create_buffer(&buffer_info, &allocation_info)
        .unwrap();

    maybe_attach_debug_label(buffer, &config.label, device);

    let start: NonNull<u8> = NonNull::new(info.mapped_data.cast()).unwrap();
    let end = NonNull::new(start.as_ptr().add(info.size.try_into().unwrap())).unwrap();
    (buffer, allocation, start, end)
}
