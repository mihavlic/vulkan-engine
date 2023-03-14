use super::*;
use crate::{
    graph::{self, allocator::round_up_pow2_usize},
    simple_handle,
};
use pumice::vk;
use std::{collections::VecDeque, marker::PhantomData, ptr::NonNull};

pub struct QueueRing {
    free_buffers: Vec<BufferEntry>,
    buffers: VecDeque<(BufferMonotonic, BufferEntry)>,
    next_id: BufferMonotonic,
    config: &'static RingConfig,
}

impl QueueRing {
    pub(crate) fn new(config: &'static RingConfig) -> Self {
        Self {
            free_buffers: Vec::new(),
            buffers: VecDeque::new(),
            next_id: BufferMonotonic(0),
            config,
        }
    }
    pub(crate) unsafe fn reset_all(&mut self) {
        self.free_buffers
            .extend(self.buffers.drain(..).map(|(_, mut b)| {
                b.cursor = b.start;
                b
            }));
    }
    pub(crate) unsafe fn reset_until_counter(&mut self, monotonic: BufferMonotonic) {
        while let Some((counter, _)) = self.buffers.front() {
            if *counter < monotonic {
                let mut b = self.buffers.pop_front().unwrap().1;
                b.cursor = b.start;
                self.free_buffers.push(b);
            } else {
                break;
            }
        }
    }
    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
        for buffer in self
            .buffers
            .iter()
            .map(|(_, b)| b)
            .chain(&self.free_buffers)
        {
            device
                .allocator()
                .destroy_buffer(buffer.buffer, buffer.allocation)
        }
    }
    pub(crate) unsafe fn head_buffer(&mut self, device: &Device) -> &mut BufferEntry {
        if self.buffers.is_empty() {
            self.add_buffer(device);
        }
        &mut self.buffers.back_mut().unwrap().1
    }
    pub(crate) unsafe fn add_fresh_head(&mut self, device: &Device) -> &mut BufferEntry {
        self.add_buffer(device);
        &mut self.buffers.back_mut().unwrap().1
    }
    fn next_monotonic(&mut self) -> BufferMonotonic {
        let next = self.next_id;
        self.next_id.0 += 1;
        next
    }
    unsafe fn add_buffer(&mut self, device: &Device) {
        let monotonic = self.next_monotonic();

        if let Some(free) = self.free_buffers.pop() {
            self.buffers.push_back((monotonic, free));
            return;
        }

        let (buffer, allocation, start, end) = make_buffer(self.config, device);

        self.buffers.push_back((
            monotonic,
            BufferEntry {
                buffer,
                allocation,
                start,
                cursor: start,
                end,
            },
        ));
    }
    pub(crate) unsafe fn allocate(
        &mut self,
        layout: std::alloc::Layout,
        device: &Device,
    ) -> SuballocatedMemory {
        assert!(layout.size() as u64 <= self.config.buffer_size);

        let mut buffer = self.head_buffer(device);

        let (ptr, dynamic_offset) = match buffer.bump(layout, device) {
            Some(ok) => ok,
            None => {
                buffer = self.add_fresh_head(device);
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
    pub(crate) fn get_monotonic_counter(&mut self) -> BufferMonotonic {
        self.next_id
    }
}
