use std::{marker::PhantomData, ptr::NonNull};

use pumice::vk;

use crate::graph::{self, allocator::round_up_pow2_usize};

use super::{debug::maybe_attach_debug_label, submission, Device};

#[derive(Clone, PartialEq, Eq)]
pub struct SuballocatedMemory {
    pub dynamic_offset: u32,
    pub buffer: vk::Buffer,
    pub memory: NonNull<u8>,
}

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
    ) -> Option<(NonNull<u8>, u32)> {
        let start = self.cursor.as_ptr() as usize;
        let aligned = round_up_pow2_usize(start, layout.align());
        let mut offset = aligned - start;

        let offset_align = device
            .physical_device_properties
            .limits
            .min_uniform_buffer_offset_alignment;
        if offset_align > 0 {
            offset = round_up_pow2_usize(offset, offset_align as usize);
        }

        let start_ptr = self.cursor.as_ptr().add(offset);
        let end_ptr = start_ptr.add(layout.size());

        if end_ptr > self.end.as_ptr() {
            return None;
        }

        self.cursor = NonNull::new(end_ptr).unwrap();

        Some((NonNull::new(start_ptr).unwrap(), offset.try_into().unwrap()))
    }
}

pub trait RingBufferCollectionConfig {
    const BUFFER_SIZE: u64;
    const USAGE: vk::BufferUsageFlags;
    const ALLOCATION_FLAGS: pumice_vma::AllocationCreateFlags;
    const REQUIRED_FLAGS: vk::MemoryPropertyFlags;
    const PREFERRED_FLAGS: vk::MemoryPropertyFlags;
    const LABEL: &'static str;
}

pub struct RingBufferCollection<C> {
    free_buffers: Vec<BufferEntry>,
    buffers: Vec<BufferEntry>,
    spooky: PhantomData<fn(C)>,
}

impl<C: RingBufferCollectionConfig> RingBufferCollection<C> {
    pub(crate) fn new() -> Self {
        Self {
            free_buffers: Vec::new(),
            buffers: Vec::new(),
            spooky: PhantomData,
        }
    }
    pub(crate) unsafe fn reset(&mut self) {
        self.free_buffers
            .extend(self.buffers.drain(..).map(|mut b| {
                b.cursor = b.start;
                b
            }));
    }
    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
        for buffer in self.buffers.iter().chain(&self.free_buffers) {
            device
                .allocator()
                .destroy_buffer(buffer.buffer, buffer.allocation)
        }
    }
    pub(crate) unsafe fn last_free_buffer(&mut self, device: &Device) -> &mut BufferEntry {
        if self.buffers.is_empty() {
            self.add_buffer(device);
        }
        self.buffers.last_mut().unwrap()
    }
    pub(crate) unsafe fn add_fresh_buffer(&mut self, device: &Device) -> &mut BufferEntry {
        self.add_buffer(device);
        self.buffers.last_mut().unwrap()
    }
    unsafe fn add_buffer(&mut self, device: &Device) {
        if let Some(free) = self.free_buffers.pop() {
            self.buffers.push(free);
            return;
        }

        let buffer_info = vk::BufferCreateInfo {
            flags: vk::BufferCreateFlags::empty(),
            size: C::BUFFER_SIZE,
            usage: C::USAGE,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let allocation_info = pumice_vma::AllocationCreateInfo {
            flags: C::ALLOCATION_FLAGS,
            usage: pumice_vma::MemoryUsage::Auto,

            ..Default::default()
        };

        let (buffer, allocation, info) = device
            .allocator()
            .create_buffer(&buffer_info, &allocation_info)
            .unwrap();

        maybe_attach_debug_label(buffer, &C::LABEL, device);

        let start: NonNull<u8> = NonNull::new(info.mapped_data.cast()).unwrap();
        let end = NonNull::new(start.as_ptr().add(info.size.try_into().unwrap())).unwrap();

        self.buffers.push(BufferEntry {
            buffer,
            allocation,
            start,
            cursor: start,
            end,
        });
    }
    pub(crate) unsafe fn allocate(
        &mut self,
        layout: std::alloc::Layout,
        device: &Device,
    ) -> SuballocatedMemory {
        assert!(layout.size() as u64 <= C::BUFFER_SIZE);

        let mut buffer = self.last_free_buffer(device);

        let (ptr, dynamic_offset) = match buffer.bump(layout, device) {
            Some(ok) => ok,
            None => {
                buffer = self.add_fresh_buffer(device);
                buffer
                    .bump(layout, device)
                    .expect("Failed to bump allocate from a fresh buffer")
            }
        };

        SuballocatedMemory {
            dynamic_offset,
            buffer: buffer.buffer,
            memory: ptr,
        }
    }
}
