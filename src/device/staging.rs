use std::{cell::UnsafeCell, collections::VecDeque, ffi::c_void, sync::Arc};

use ahash::HashSet;
use bumpalo::Bump;
use pumice::{util::ObjectHandle, vk};
use smallvec::SmallVec;

use crate::{
    device::debug::{with_temporary_cstr, LazyDisplay},
    graph::{
        compile::{CombinedResourceHandle, ResourceFirstAccessInterface},
        execute::{
            do_dummy_submissions, handle_imported_sync_generic, DummySubmission, SubmissionExtra,
        },
        resource_marker::{TypeNone, TypeOption},
        task::UnsafeSend,
        GraphBuffer, GraphImage,
    },
    object::{self, HostAccessKind},
    storage::constant_ahash_hashset,
};

use super::{
    ringbuffer_collection::{RingBufferCollection, RingBufferCollectionConfig},
    submission::{self, QueueSubmission, SubmissionManager, TimelineSemaphore},
    Device,
};

struct BufferFlushRange {
    allocation: pumice_vma::Allocation,
    offset: u64,
    size: u64,
}

struct StagingManagerConfig;
impl RingBufferCollectionConfig for StagingManagerConfig {
    const BUFFER_SIZE: u64 = 16 * 1024 * 1024;
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags(
        vk::BufferUsageFlags::TRANSFER_SRC.0 | vk::BufferUsageFlags::TRANSFER_DST.0,
    );
    const ALLOCATION_FLAGS: pumice_vma::AllocationCreateFlags = pumice_vma::AllocationCreateFlags(
        pumice_vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE.0
            | pumice_vma::AllocationCreateFlags::MAPPED.0,
    );
    const REQUIRED_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_COHERENT;
    const PREFERRED_FLAGS: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::empty();
    const LABEL: &'static str = "DescriptorAllocator buffer";
}

pub struct StagingManager {
    queue: submission::Queue,
    buffers: RingBufferCollection<StagingManagerConfig>,
    command: CommandBufferPool,
}

impl StagingManager {
    pub(crate) unsafe fn new(transfer_queue: submission::Queue, device: &Device) -> Self {
        Self {
            queue: transfer_queue,
            buffers: RingBufferCollection::new(),
            command: CommandBufferPool::new(transfer_queue.family, device),
        }
    }
    pub unsafe fn copy_to_buffer<F: FnMut(*mut u8) + Send + 'static>(
        &mut self,
        dst_buffer: object::Buffer,
        offset: u64,
        layout: std::alloc::Layout,
        mut fun: F,
        device: &Device,
    ) {
        let allocator = device.allocator();
        let allocation = dst_buffer.get_allocation();

        let (cmd, submission, finalizers) =
            self.command.get_current_command_buffer(self.queue, device);

        dst_buffer.access_mutable(
            |d| d.get_mutable_state(),
            |s| {
                // if the destination buffer is mappable, we'll write to it directly
                if let Ok(ptr) = allocator.map_memory(allocation) {
                    s.synchronization_state().update_host_access(|prev_access| {
                        if prev_access.is_empty() {
                            fun(ptr);
                            allocator.flush_allocation(allocation, offset, layout.size() as u64);
                            HostAccessKind::Immediate
                        } else {
                            let (submission, semaphore) = device.make_submission(None);
                            let ptr = unsafe { UnsafeSend::new(ptr) };
                            let group = device.open_generation_finalized(move |device| {
                                fun(*ptr);
                                let signal_info = vk::SemaphoreSignalInfoKHR {
                                    semaphore: semaphore.raw,
                                    value: semaphore.value,
                                    ..Default::default()
                                };
                                device.device().signal_semaphore_khr(&signal_info).unwrap();
                            });

                            group.add_submissions(prev_access.iter().copied());
                            group.finish();

                            HostAccessKind::Synchronized(submission)
                        }
                    });

                    allocator.unmap_memory(allocation);
                }
                // otherwise schedule a transfer operation
                else {
                    let staging = self.buffers.allocate(layout, device);
                    fun(staging.memory.as_ptr());

                    let sharing_mode_concurrent =
                        dst_buffer.get_create_info().sharing_mode_concurrent;

                    let result = s.synchronization_state().update(
                        self.queue.family(),
                        TypeNone::new_none(),
                        &[submission],
                        TypeNone::new_none(),
                        self.queue.family(),
                        sharing_mode_concurrent,
                    );

                    assert!(result.transition_ownership_from.is_none(), "TODO");

                    // struct A(u32);
                    // impl ResourceFirstAccessInterface for A {
                    //     type Accessor = ();
                    //     fn accessors(&self) -> &[Self::Accessor] {
                    //         &[()]
                    //     }
                    //     fn layout(&self) -> vk::ImageLayout {
                    //         vk::ImageLayout::UNDEFINED
                    //     }
                    //     fn stages(&self) -> vk::PipelineStageFlags2KHR {
                    //         vk::PipelineStageFlags2KHR::COPY_KHR
                    //     }
                    //     fn access(&self) -> vk::AccessFlags2KHR {
                    //         vk::AccessFlags2KHR::TRANSFER_WRITE_KHR
                    //     }
                    //     fn queue_family(&self) -> u32 {
                    //         self.0
                    //     }
                    // }

                    // let mut extra = SubmissionExtra {
                    //     image_barriers: Vec::new(),
                    //     buffer_barriers: Vec::new(),
                    //     dependencies: constant_ahash_hashset(),
                    // };

                    // let mut dummy_submissions = Vec::new();

                    // handle_imported_sync_generic(
                    //     CombinedResourceHandle::new_buffer(GraphBuffer::new(0)),
                    //     sharing_mode_concurrent,
                    //     &A(self.queue.family()),
                    //     &[submission],
                    //     TypeNone::new_none(),
                    //     self.queue.family(),
                    //     s.synchronization_state(),
                    //     &mut extra,
                    //     |d| dummy_submissions.push(d),
                    //     |_, m| m,
                    //     device,
                    // );

                    let d = device.device();
                    if device.debug() {
                        let display = LazyDisplay(|f| {
                            if let Some(label) = dst_buffer.get_create_info().label.as_ref() {
                                write!(f, "Copy to '{label}'")
                            } else {
                                write!(f, "Copy to buffer {:p}", dst_buffer.get_handle())
                            }
                        });

                        with_temporary_cstr(&display, |cstr| {
                            let info = vk::DebugUtilsLabelEXT {
                                p_label_name: cstr,
                                ..Default::default()
                            };
                            d.cmd_begin_debug_utils_label_ext(cmd, &info);
                        });
                    }

                    d.cmd_copy_buffer(
                        cmd,
                        staging.buffer,
                        dst_buffer.get_handle(),
                        &[vk::BufferCopy {
                            src_offset: staging.dynamic_offset as u64,
                            dst_offset: offset,
                            size: layout.size() as u64,
                        }],
                    );

                    // assert!(extra.image_barriers.is_empty());

                    // let bump = Bump::new();
                    // let mut submit_infos = Vec::<vk::SubmitInfo2>::new();
                    // let mut raw_memory_barriers = Vec::<vk::MemoryBarrier2>::new();
                    // let mut raw_image_barriers = Vec::<vk::ImageMemoryBarrier2>::new();
                    // let mut raw_buffer_barriers = Vec::<vk::BufferMemoryBarrier2>::new();

                    // do_dummy_submissions(
                    //     &bump,
                    //     &dummy_submissions,
                    //     &mut submit_infos,
                    //     &mut raw_memory_barriers,
                    //     &mut raw_image_barriers,
                    //     &mut raw_buffer_barriers,
                    //     &[],
                    //     &[Some(dst_buffer.get_handle())],
                    //     device,
                    //     |_| unreachable!(),
                    //     |_, _| unreachable!(),
                    //     |queue_family, device| todo!(),
                    // );

                    // let buffer_barriers = extra
                    //     .buffer_barriers
                    //     .iter()
                    //     .map(|b| b.to_vk(dst_buffer.get_handle()));
                    // let memory_barriers = extra.dependencies.iter().map(|s| s.s)

                    if device.debug() {
                        d.cmd_end_debug_utils_label_ext(cmd);
                    }
                }
            },
        );
    }
}

#[derive(Clone)]
enum CommandBufferState {
    Empty,
    Dirty {
        submission: QueueSubmission,
        semaphore: TimelineSemaphore,
        // finalizers: UnsafeSend<Arc<UnsafeCell<SmallVec<[Box<dyn FnOnce(&Device) + Send>; 2]>>>>,
        dirty: bool,
    },
}

struct CommandBufferPool {
    pool: vk::CommandPool,
    buffers: VecDeque<(QueueSubmission, vk::CommandBuffer)>,
    free_buffers: Vec<(CommandBufferState, vk::CommandBuffer)>,
}

impl CommandBufferPool {
    unsafe fn new(queue_family: u32, device: &Device) -> Self {
        let info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::TRANSIENT
                | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: queue_family,
            ..Default::default()
        };

        Self {
            pool: device
                .device()
                .create_command_pool(&info, device.allocator_callbacks())
                .unwrap(),
            buffers: VecDeque::new(),
            free_buffers: Vec::new(),
        }
    }
    pub(crate) fn poll(&mut self, submission_manager: &SubmissionManager) {
        while let Some(&(submission, buffer)) = self.buffers.front() {
            if submission_manager.is_submission_finished(submission) {
                self.buffers.pop_front();
                self.free_buffers.push((CommandBufferState::Empty, buffer));
            } else {
                break;
            }
        }
    }
    unsafe fn get_current_command_buffer(
        &mut self,
        queue: submission::Queue,
        device: &Device,
    ) -> (
        vk::CommandBuffer,
        QueueSubmission,
        // &mut SmallVec<[Box<dyn FnOnce(&Device) + Send>; 2]>,
        &mut bool,
    ) {
        if self.free_buffers.is_empty() {
            self.add_buffer(device);
        }

        let (state, buffer) = self.free_buffers.last_mut().unwrap();
        if let CommandBufferState::Empty = state {
            // let finalizers: UnsafeSend<
            //     Arc<UnsafeCell<SmallVec<[Box<dyn FnOnce(&Device) + Send>; 2]>>>,
            // > = unsafe { UnsafeSend::new(Arc::new(UnsafeCell::new(SmallVec::new()))) };
            // let finalizer_finalizer = {
            //     let finalizers_copy = finalizers.clone();
            //     move |device: &Device| {
            //         let arc = UnsafeSend::take(finalizers_copy);
            //         let unsafe_cell = Arc::try_unwrap(arc).ok().unwrap();
            //         for finalizer in UnsafeCell::into_inner(unsafe_cell) {
            //             finalizer(device);
            //         }
            //     }
            // };

            let (submission, semaphore) =
                device.make_submission(None /* Some(Box::new(finalizer_finalizer)) */);

            *state = CommandBufferState::Dirty {
                submission,
                semaphore,
                dirty: false,
            };

            device
                .device()
                .begin_command_buffer(
                    *buffer,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap();
        }

        let CommandBufferState::Dirty {
                submission,
                dirty,
                ..
            } = state else {
                unreachable!()
            };

        (*buffer, *submission, dirty)
    }
    unsafe fn add_buffer(&mut self, device: &Device) {
        let info = vk::CommandBufferAllocateInfo {
            command_pool: self.pool,
            level: vk::CommandBufferLevel::PRIMARY,
            // TODO allocate more buffers at a time?
            command_buffer_count: 1,
            ..Default::default()
        };
        let buffer = device.device().allocate_command_buffers(&info).unwrap()[0];
        self.free_buffers.push((CommandBufferState::Empty, buffer));
    }
    unsafe fn flush(
        &mut self,
        queue: submission::Queue,
        device: &Device,
    ) -> Option<QueueSubmission> {
        if let Some((CommandBufferState::Dirty { dirty: true, .. }, _)) = self.free_buffers.last() {
            let Some((CommandBufferState::Dirty { submission, semaphore, dirty }, buffer)) = self.free_buffers.pop() else {
                unreachable!()
            };

            self.buffers.push_back((submission, buffer));

            let signal = vk::SemaphoreSubmitInfoKHR {
                semaphore: semaphore.raw,
                value: semaphore.value,
                stage_mask: vk::PipelineStageFlags2KHR::COPY,
                ..Default::default()
            };

            let buffer_info = vk::CommandBufferSubmitInfoKHR {
                command_buffer: buffer,
                ..Default::default()
            };

            let submit_info = vk::SubmitInfo2KHR {
                flags: vk::SubmitFlagsKHR::empty(),
                command_buffer_info_count: 1,
                p_command_buffer_infos: &buffer_info,
                signal_semaphore_info_count: 1,
                p_signal_semaphore_infos: &signal,
                ..Default::default()
            };

            device.device().end_command_buffer(buffer);

            device
                .device()
                .queue_submit_2_khr(queue.raw(), &[submit_info], vk::Fence::null())
                .unwrap();

            return Some(submission);
        }

        None
    }
    unsafe fn destroy(&mut self, device: &Device) {
        device
            .device()
            .destroy_command_pool(self.pool, device.allocator_callbacks());
    }
}
