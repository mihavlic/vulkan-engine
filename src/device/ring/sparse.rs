use super::*;
use crate::{
    device::{submission::SubmissionManager, QueueSubmission},
    graph::{self, allocator::round_up_pow2_usize},
    simple_handle,
};
use pumice::vk;
use smallvec::SmallVec;
use std::{collections::VecDeque, hash::Hash, marker::PhantomData, ptr::NonNull, sync::Arc};

struct SparseBufferEntry {
    submissions: SmallVec<[QueueSubmission; 4]>,
    entry: BufferEntry,
}

pub struct SparseRing {
    free_buffers: Vec<BufferEntry>,
    buffers: Vec<SparseBufferEntry>,
    config: &'static RingConfig,
}

impl SparseRing {
    pub(crate) fn new(config: &'static RingConfig) -> Self {
        Self {
            free_buffers: Vec::new(),
            buffers: Vec::new(),
            config,
        }
    }
    pub(crate) fn collect(&mut self, submissions: &SubmissionManager) {
        self.buffers.retain_mut(|b| {
            b.submissions
                .retain(|s| !submissions.is_submission_finished(*s));

            if b.submissions.is_empty() {
                let mut buffer = b.entry.clone();
                buffer.cursor = buffer.start;
                self.free_buffers.push(buffer);

                false
            } else {
                true
            }
        });
    }
    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
        for buffer in self
            .buffers
            .drain(..)
            .map(|b| b.entry)
            .chain(self.free_buffers.drain(..))
        {
            device
                .allocator()
                .destroy_buffer(buffer.buffer, buffer.allocation)
        }
    }
    unsafe fn head_buffer(&mut self, device: &Device) -> &mut SparseBufferEntry {
        if self.buffers.is_empty() {
            self.add_buffer(device);
        }
        self.buffers.last_mut().unwrap()
    }
    unsafe fn add_fresh_head(&mut self, device: &Device) -> &mut SparseBufferEntry {
        self.add_buffer(device);
        self.buffers.last_mut().unwrap()
    }
    unsafe fn add_buffer(&mut self, device: &Device) {
        if let Some(entry) = self.free_buffers.pop() {
            self.buffers.push(SparseBufferEntry {
                submissions: SmallVec::new(),
                entry,
            });
            return;
        }

        let (buffer, allocation, start, end) = make_buffer(self.config, device);

        self.buffers.push(SparseBufferEntry {
            entry: BufferEntry {
                buffer,
                allocation,
                start,
                cursor: start,
                end,
            },
            submissions: SmallVec::new(),
        });
    }
    pub(crate) unsafe fn allocate(
        &mut self,
        layout: std::alloc::Layout,
        submission: QueueSubmission,
        device: &Device,
    ) -> SuballocatedMemory {
        assert!(layout.size() as u64 <= self.config.buffer_size);

        let mut buffer = self.head_buffer(device);

        let (ptr, buffer_offset) = match buffer.entry.bump(layout, device) {
            Some(ok) => ok,
            None => {
                buffer = self.add_fresh_head(device);
                buffer
                    .entry
                    .bump(layout, device)
                    .expect("Failed to bump allocate from a fresh buffer")
            }
        };

        // this has poor scaling behaviour, but we expect only a small number of elements
        if !buffer.submissions.contains(&submission) {
            buffer.submissions.push(submission);
        }

        SuballocatedMemory {
            buffer_offset: buffer_offset,
            buffer: buffer.entry.buffer,
            memory: ptr,
        }
    }
}
