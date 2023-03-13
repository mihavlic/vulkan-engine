use std::{
    cell::{Cell, UnsafeCell},
    collections::VecDeque,
    ffi::c_void,
    sync::Arc,
};

use ahash::{AHashSet, HashSet};
use bumpalo::Bump;
use pumice::{util::ObjectHandle, vk, DeviceWrapper};
use smallvec::SmallVec;

use crate::{
    device::debug::{with_temporary_cstr, LazyDisplay},
    graph::{
        compile::{CombinedResourceHandle, ResourceFirstAccessInterface},
        execute::{
            do_dummy_submissions, handle_imported_sync_generic, DummySubmission, SubmissionExtra,
        },
        resource_marker::{TypeNone, TypeOption, TypeSome},
        task::UnsafeSend,
        GraphBuffer, GraphImage,
    },
    object::{self, HostAccessKind, ObjRef},
    storage::constant_ahash_hashset,
    util::ffi_ptr::AsFFiPtr,
};

use super::{
    debug::debug_label_span,
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

// #[repr(align(8))]
// pub struct Align8(u8);

// impl Align8 {
//     #[inline]
//     fn new(bytes: *const u8) -> Option<*const Align8> {
//         ((bytes as usize & 7) == 0).then_some(bytes.cast())
//     }
//     #[inline]
//     fn new_mut(bytes: *mut u8) -> Option<*mut Align8> {
//         ((bytes as usize & 7) == 0).then_some(bytes.cast())
//     }
// }

// pub trait AlignedBytes {
//     fn bytes(self) -> *const u8;
// }

// pub trait AlignedBytesMut {
//     fn bytes_mut(self) -> *mut u8;
// }

// impl AlignedBytes for *const Align8 {
//     #[inline(always)]
//     fn bytes(self) -> *const u8 {
//         self.cast::<u8>()
//     }
// }

// impl AlignedBytesMut for *mut Align8 {
//     #[inline(always)]
//     fn bytes_mut(self) -> *mut u8 {
//         self.cast::<u8>()
//     }
// }

pub struct ImageWrite {
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_subresource: vk::ImageSubresourceLayers,
    pub image_offset: vk::Offset3D,
    pub image_extent: vk::Extent3D,
}

unsafe impl Send for StagingManager {}
unsafe impl Sync for StagingManager {}
impl StagingManager {
    pub(crate) unsafe fn new(
        transfer_queue: submission::Queue,
        callbacks: Option<&vk::AllocationCallbacks>,
        device: &DeviceWrapper,
    ) -> Self {
        Self {
            queue: transfer_queue,
            buffers: RingBufferCollection::new(),
            command: CommandBufferPool::new(transfer_queue.family, callbacks, device),
        }
    }
    pub unsafe fn write_buffer<F: FnMut(*mut u8) + Send + 'static>(
        &mut self,
        dst_buffer: &ObjRef<object::Buffer>,
        offset: u64,
        layout: std::alloc::Layout,
        mut fun: F,
        device: &Device,
    ) {
        if layout.size() == 0 {
            return;
        }

        let allocator = device.allocator();
        let allocation = dst_buffer.get_allocation();

        let allocation_info = allocator.get_allocation_info(allocation);

        let (cmd, submission, wait_submissions, dirty) =
            self.command.get_current_command_buffer(self.queue, device);

        let mut fun = Some(fun);
        let synchronization_result = dst_buffer.access_mutable(
            |d| d.get_mutable_state(),
            |s| {
                // if the destination buffer is mapped, we'll write to it directly
                if let Ok(ptr) = allocator.map_memory(allocation) {
                    assert!(
                        (ptr as usize & (layout.align() - 1)) == 0,
                        "Buffer allocation is not properly aligned"
                    );

                    let mapped_size = layout.size();
                    let mut fun = fun.take().unwrap();

                    s.synchronization_state().update_host_access(|prev_access| {
                        if prev_access.is_empty() {
                            fun(ptr);

                            allocator.flush_allocation(allocation, offset, mapped_size as u64);
                            allocator.unmap_memory(allocation);

                            HostAccessKind::Immediate
                        } else {
                            let (submission, semaphore) = device.make_submission(None);
                            let ptr = unsafe { UnsafeSend::new(ptr) };
                            let group = device.open_generation_finalized(move |device| {
                                fun(*ptr);

                                device.allocator().flush_allocation(
                                    allocation,
                                    offset,
                                    mapped_size as u64,
                                );
                                device.allocator().unmap_memory(allocation);

                                let signal_info = vk::SemaphoreSignalInfoKHR {
                                    semaphore: semaphore.raw,
                                    value: semaphore.value,
                                    ..Default::default()
                                };
                                device.device().signal_semaphore_khr(&signal_info).unwrap();
                            });

                            group.add_submissions(prev_access.iter().copied());
                            group.finish();

                            wait_submissions.insert(submission);
                            HostAccessKind::Synchronized(submission)
                        }
                    });

                    None
                }
                // otherwise schedule a transfer operation
                else {
                    let result = s.synchronization_state().update(
                        self.queue.family(),
                        TypeNone::new_none(),
                        &[submission],
                        TypeNone::new_none(),
                        self.queue.family(),
                        dst_buffer.get_create_info().sharing_mode_concurrent,
                    );

                    Some(result)
                }
            },
        );

        let Some(result) = synchronization_result else {
            return;
        };

        assert!(result.transition_ownership_from.is_none(), "TODO");

        *dirty = true;
        wait_submissions.extend(result.prev_access);
        let mut fun = fun.take().unwrap();

        let staging = self.buffers.allocate(layout, device);
        fun(staging.memory.as_ptr());

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
        // )

        let d = device.device();
        debug_label_span(
            cmd,
            device,
            |f| {
                if let Some(label) = dst_buffer.get_create_info().label.as_ref() {
                    write!(f, "Copy to '{label}'")
                } else {
                    write!(f, "Copy to buffer {:p}", dst_buffer.get_handle())
                }
            },
            || {
                if let Some(from) = result.transition_layout_from {
                    let barrier = vk::ImageMemoryBarrier2KHR {
                        dst_stage_mask: vk::PipelineStageFlags2KHR::COPY,
                        dst_access_mask: vk::AccessFlags2KHR::TRANSFER_WRITE,
                        old_layout: from,
                        new_layout: todo!(),
                        image: todo!(),
                        subresource_range: todo!(),
                        ..Default::default()
                    };
                    let dependency_info = vk::DependencyInfoKHR {
                        image_memory_barrier_count: 1,
                        p_image_memory_barriers: &barrier,
                        ..Default::default()
                    };
                    d.cmd_pipeline_barrier_2_khr(cmd, &dependency_info);
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
            },
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
    }
    pub unsafe fn write_image<F: FnMut(*mut u8, usize, &ImageWrite) + Send + 'static>(
        &mut self,
        dst_image: &ObjRef<object::Image>,
        texel_layout: std::alloc::Layout,
        regions: Vec<ImageWrite>,
        mut fun: F,
        device: &Device,
    ) {
        if regions.is_empty() {
            return;
        }
        assert!(
            texel_layout.size() & (texel_layout.align() - 1) == 0,
            "Size is not a multiple of alignment"
        );

        let raw_image = dst_image.get_handle();
        let allocator = device.allocator();
        let allocation = dst_image.get_allocation();

        let (cmd, submission, wait_submissions, dirty) =
            self.command.get_current_command_buffer(self.queue, device);

        // TODO we may be able to skip the transfer image copy
        // if the destination image is host accessible and has linear tiling
        // and manually do the copy through a mapped pointer
        // for now we don't do that since such images will be very rare
        let result = dst_image.access_mutable(
            |d| d.get_mutable_state(),
            |s| {
                let result = s.synchronization_state().update(
                    self.queue.family(),
                    TypeSome::new_some(vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                    &[submission],
                    TypeSome::new_some(vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                    self.queue.family(),
                    dst_image.get_create_info().sharing_mode_concurrent,
                );

                result
            },
        );

        *dirty = true;
        wait_submissions.extend(result.prev_access);
        assert!(result.transition_ownership_from.is_none(), "TODO");

        let region_sources = regions.iter().enumerate().map(|(i, region)| {
            let mut width = region.buffer_row_length;
            if width == 0 {
                width = region.image_extent.width;
            }

            let mut height = region.buffer_image_height;
            if height == 0 {
                height = region.image_extent.height;
            }

            let depth = region.image_extent.depth;

            let texels = width * height * depth;
            assert!(texels > 0);

            let layout = std::alloc::Layout::from_size_align(
                texels as usize * texel_layout.size(),
                texel_layout.align(),
            )
            .unwrap();
            let staging = self.buffers.allocate(layout, device);

            fun(staging.memory.as_ptr(), i, region);

            (staging.buffer, staging.dynamic_offset, region)
        });

        let d = device.device();

        let mut converted_regions: SmallVec<[_; 8]> = SmallVec::new();
        let mut last_buffer = None;
        let mut convert_region = |(buffer, offset, region): (vk::Buffer, u32, &ImageWrite)| {
            // if the source buffer changed (due to the last ringbuffer chunk getting filled)
            // we need to split the copy into multiple calls
            if let Some(src_buffer) = last_buffer {
                debug_assert!(src_buffer != vk::Buffer::null());
                if src_buffer != buffer {
                    debug_assert!(!converted_regions.is_empty());
                    d.cmd_copy_buffer_to_image(
                        cmd,
                        src_buffer,
                        raw_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &converted_regions,
                    );
                    converted_regions.clear();
                }
            }

            if buffer != vk::Buffer::null() {
                converted_regions.push(vk::BufferImageCopy {
                    buffer_offset: offset.try_into().unwrap(),
                    buffer_row_length: region.buffer_row_length,
                    buffer_image_height: region.buffer_image_height,
                    image_subresource: region.image_subresource.clone(),
                    image_offset: region.image_offset.clone(),
                    image_extent: region.image_extent.clone(),
                });
            }
            last_buffer = Some(buffer);
        };

        debug_label_span(
            cmd,
            device,
            |f| {
                if let Some(label) = dst_image.get_create_info().label.as_ref() {
                    write!(f, "Copy to '{label}'")
                } else {
                    write!(f, "Copy to image {:p}", dst_image.get_handle())
                }
            },
            || {
                if let Some(from) = result.transition_layout_from {
                    let barrier = vk::ImageMemoryBarrier2KHR {
                        dst_stage_mask: vk::PipelineStageFlags2KHR::COPY,
                        dst_access_mask: vk::AccessFlags2KHR::TRANSFER_WRITE,
                        old_layout: from,
                        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        image: raw_image,
                        // we synchronize resources like a mutex
                        // no partial overlapping access is allowed
                        subresource_range: dst_image.get_whole_subresource_range(),
                        ..Default::default()
                    };
                    let dependency_info = vk::DependencyInfoKHR {
                        image_memory_barrier_count: 1,
                        p_image_memory_barriers: &barrier,
                        ..Default::default()
                    };
                    d.cmd_pipeline_barrier_2_khr(cmd, &dependency_info);
                }

                for r in region_sources {
                    convert_region(r);
                }
                // flush the last buffer with a null buffer (specially handled)
                convert_region((vk::Buffer::null(), 0, regions.first().unwrap()));
            },
        );
    }
    pub(crate) unsafe fn flush(&mut self, device: &Device) -> Option<QueueSubmission> {
        self.command.flush(self.queue, device)
    }
    pub(crate) fn collect(&mut self, submission_manager: &SubmissionManager) {
        self.command.poll(submission_manager)
    }
}

pub enum WriteCommand<'a> {
    Buffer {
        buffer: &'a ObjRef<object::Buffer>,
        offset: u64,
        layout: std::alloc::Layout,
        callback: Box<dyn FnMut(*mut u8) + Send>,
    },
    Image {
        image: &'a ObjRef<object::Image>,
        texel_layout: std::alloc::Layout,
        regions: Vec<ImageWrite>,
        callback: Box<dyn FnMut(*mut u8, usize, &ImageWrite) + Send>,
    },
}

impl Device {
    pub unsafe fn write_buffer<F: FnMut(*mut u8) + Send + 'static>(
        &self,
        buffer: &ObjRef<object::Buffer>,
        offset: u64,
        layout: std::alloc::Layout,
        mut fun: F,
    ) {
        self.staging_manager
            .write()
            .write_buffer(buffer, offset, layout, fun, self);
    }
    pub unsafe fn write_image<F: FnMut(*mut u8, usize, &ImageWrite) + Send + 'static>(
        &self,
        image: &ObjRef<object::Image>,
        texel_layout: std::alloc::Layout,
        regions: Vec<ImageWrite>,
        mut fun: F,
    ) {
        self.staging_manager
            .write()
            .write_image(image, texel_layout, regions, fun, self)
    }
    pub unsafe fn flush_staged_transfers(&self) -> Option<QueueSubmission> {
        self.staging_manager.write().flush(self)
    }
    pub unsafe fn write_multiple<'a>(
        &self,
        commands: impl IntoIterator<Item = WriteCommand<'a>>,
    ) -> Option<QueueSubmission> {
        let mut staging = None;

        for c in commands {
            let staging = staging.get_or_insert_with(|| self.staging_manager.write());

            match c {
                WriteCommand::Buffer {
                    buffer,
                    offset,
                    layout,
                    callback,
                } => staging.write_buffer(buffer, offset, layout, callback, self),
                WriteCommand::Image {
                    image,
                    texel_layout,
                    regions,
                    callback,
                } => staging.write_image(image, texel_layout, regions, callback, self),
            }
        }

        staging?.flush(self)
    }
}

#[derive(Clone)]
enum CommandBufferState {
    Empty,
    Dirty {
        wait_submissions: ahash::HashSet<QueueSubmission>,
        submission: QueueSubmission,
        semaphore: TimelineSemaphore,
        dirty: bool,
    },
}

struct CommandBufferPool {
    pool: vk::CommandPool,
    buffers: VecDeque<(QueueSubmission, vk::CommandBuffer)>,
    free_buffers: Vec<(CommandBufferState, vk::CommandBuffer)>,
}

impl CommandBufferPool {
    unsafe fn new(
        queue_family: u32,
        callbacks: Option<&vk::AllocationCallbacks>,
        device: &DeviceWrapper,
    ) -> Self {
        let info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::TRANSIENT
                | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: queue_family,
            ..Default::default()
        };

        Self {
            pool: device.create_command_pool(&info, callbacks).unwrap(),
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
        &mut ahash::HashSet<QueueSubmission>,
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
                wait_submissions: constant_ahash_hashset(),
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
            wait_submissions,
            ..
        } = state else {
            unreachable!()
        };

        (*buffer, *submission, wait_submissions, dirty)
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
            let Some((CommandBufferState::Dirty { submission, semaphore, dirty, wait_submissions }, buffer)) = self.free_buffers.pop() else {
                unreachable!()
            };

            self.buffers.push_back((submission, buffer));

            let waits: SmallVec<[_; 8]> = {
                assert!(
                    !wait_submissions.contains(&submission),
                    "Cyclic dependency when submitting staging work"
                );

                let mut active: SmallVec<[_; 8]> = SmallVec::new();
                device.collect_active_submission_datas(wait_submissions, &mut active);

                active
                    .iter()
                    .map(|semaphore| vk::SemaphoreSubmitInfoKHR {
                        semaphore: semaphore.raw,
                        value: semaphore.value,
                        stage_mask: vk::PipelineStageFlags2KHR::ALL_COMMANDS,
                        ..Default::default()
                    })
                    .collect()
            };

            let signal = vk::SemaphoreSubmitInfoKHR {
                semaphore: semaphore.raw,
                value: semaphore.value,
                stage_mask: vk::PipelineStageFlags2KHR::TRANSFER,
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
                wait_semaphore_info_count: waits.len() as u32,
                p_wait_semaphore_infos: waits.as_ffi_ptr(),
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
