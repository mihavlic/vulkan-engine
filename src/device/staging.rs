use std::{
    borrow::Cow,
    cell::{Cell, UnsafeCell},
    collections::VecDeque,
    ffi::c_void,
    future::Future,
    mem::ManuallyDrop,
    sync::{Arc, OnceLock},
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
        task::{SendSyncUnsafeCell, UnsafeSend, UnsafeSendSync},
        GraphBuffer, GraphImage,
    },
    object::{self, HostAccessKind, ObjRef},
    storage::{constant_ahash_hashset, DefaultAhashMap, DefaultAhashSet},
    util::ffi_ptr::AsFFiPtr,
};

use super::{
    debug::debug_label_span,
    maybe_attach_debug_label,
    ring::{QueueRing, RingConfig},
    submission::{self, QueueSubmission, SubmissionManager, TimelineSemaphore},
    Device, SparseRing,
};

struct BufferFlushRange {
    allocation: pumice_vma::Allocation,
    offset: u64,
    size: u64,
}

const CONFIG: RingConfig = RingConfig {
    buffer_size: 64 * 1024 * 1024,
    usage: vk::BufferUsageFlags(
        vk::BufferUsageFlags::TRANSFER_SRC.0 | vk::BufferUsageFlags::TRANSFER_DST.0,
    ),
    allocation_flags: vma::AllocationCreateFlags(
        vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE.0
            | vma::AllocationCreateFlags::MAPPED.0,
    ),
    required_flags: vk::MemoryPropertyFlags::HOST_COHERENT,
    preferred_flags: vk::MemoryPropertyFlags::empty(),
    label: "DescriptorAllocator buffer",
};

pub struct StagingManager {
    queue: submission::Queue,
    ring: SparseRing,
    command: CommandBufferPool,
}

pub struct ImageRegion {
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_subresource: vk::ImageSubresourceLayers,
    pub image_offset: vk::Offset3D,
    pub image_extent: vk::Extent3D,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ResourceAccessStatus {
    /// Some shortcircuiting behaviour caused no access to occur, something probably had a size of zero
    #[default]
    None,
    /// The resource is host accessible and had no active accessors, the access was performed immediatelly
    Immediate,
    /// The resource is host accessible but has currently active accessors, a generation was opened to track
    /// its availability. The access will be performed as the generations finalizer
    HostDelayed,
    /// The resource is not host accessible, a device operation was recorded into a command buffer
    Recorded,
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
            ring: SparseRing::new(&CONFIG),
            command: CommandBufferPool::new(
                transfer_queue.family,
                callbacks,
                Some("StagingManager".into()),
                device,
            ),
        }
    }
    /// If ResourceAccessStatus::Recorded is returned, the tuple contains a Some of the passed in callback
    unsafe fn read_buffer_impl(
        &mut self,
        buffer: &ObjRef<object::Buffer>,
        offset: usize,
        layout: std::alloc::Layout,

        cmd: vk::CommandBuffer,
        queue_family: u32,
        submission: QueueSubmission,

        check_availability: bool,
        device: &Device,

        fun: Box<dyn FnOnce(*const u8) + Send + 'static>,
    ) -> (
        ResourceAccessStatus,
        Option<(Box<dyn FnOnce(*const u8) + Send + 'static>, *const u8)>,
        SmallVec<[QueueSubmission; 4]>,
    ) {
        if layout.size() == 0 {
            return (ResourceAccessStatus::None, None, SmallVec::new());
        }

        let allocator = device.allocator();
        let allocation = buffer.get_allocation();
        let info = allocator.get_allocation_info(allocation);
        let mems = allocator.get_memory_properties();

        let allocation_info = allocator.get_allocation_info(allocation);

        let mut submissions = SmallVec::new();
        let mut status = ResourceAccessStatus::Recorded;
        let mut fun = Some(fun);

        let synchronization_result = buffer.access_mutable(
            |d| d.get_mutable_state(),
            |s| {
                // if the destination buffer is mappable, we'll write to it directly
                if mems.memory_types[info.memory_type as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                {
                    let ptr = allocator.map_memory(allocation).unwrap();
                    let ptr = ptr.add(offset);

                    assert!(
                        (ptr as usize & (layout.align() - 1)) == 0,
                        "Buffer allocation is not properly aligned"
                    );

                    let mapped_size = layout.size();
                    let mut fun = fun.take().unwrap();

                    let synchronization_state = &mut s.synchronization_state();

                    if check_availability {
                        synchronization_state.retain_active_submissions(device);
                    }

                    synchronization_state.update_host_access(|prev_access| {
                        if prev_access.is_empty() {
                            allocator.invalidate_allocation(
                                allocation,
                                offset as u64,
                                mapped_size as u64,
                            );

                            fun(ptr);

                            allocator.unmap_memory(allocation);

                            status = ResourceAccessStatus::Immediate;
                            HostAccessKind::Immediate
                        } else {
                            let (submission, semaphore) = device.make_submission(None);
                            let ptr = unsafe { UnsafeSend::new(ptr) };
                            let group = device.open_generation_finalized(move |device| {
                                device.allocator().invalidate_allocation(
                                    allocation,
                                    offset as u64,
                                    mapped_size as u64,
                                );

                                fun(*ptr);

                                device.allocator().unmap_memory(allocation);

                                let signal_info = vk::SemaphoreSignalInfoKHR {
                                    semaphore: semaphore.raw,
                                    value: semaphore.value,
                                    ..Default::default()
                                };
                                device.device().signal_semaphore_khr(&signal_info).unwrap();
                            });

                            group.add_submissions(prev_access.iter().copied());

                            submissions.push(submission);

                            status = ResourceAccessStatus::HostDelayed;
                            HostAccessKind::Synchronized(submission)
                        }
                    });

                    None
                }
                // otherwise schedule a transfer operation
                else {
                    let result = s.synchronization_state().update(
                        queue_family,
                        TypeNone::new_none(),
                        &[submission],
                        TypeNone::new_none(),
                        queue_family,
                        buffer.get_create_info().sharing_mode_concurrent,
                    );

                    status = ResourceAccessStatus::Recorded;
                    Some(result)
                }
            },
        );

        let result = match status {
            ResourceAccessStatus::None => unreachable!(),
            ResourceAccessStatus::Immediate | ResourceAccessStatus::HostDelayed => {
                return (status, None, submissions);
            }
            ResourceAccessStatus::Recorded => {
                assert!(submissions.is_empty());
                synchronization_result.unwrap()
            }
        };

        assert!(result.transition_ownership_from.is_none(), "TODO");
        assert!(result.transition_layout_from.is_none(), "unreachable");

        let mut fun = fun.take().unwrap();

        let staging = self.ring.allocate(layout, submission, device);

        let d = device.device();
        debug_label_span(
            cmd,
            device,
            |f| {
                if let Some(label) = buffer.get_create_info().label.as_ref() {
                    write!(f, "Read '{label}'")
                } else {
                    write!(f, "Read buffer {:p}", buffer.get_handle())
                }
            },
            || {
                d.cmd_copy_buffer(
                    cmd,
                    buffer.get_handle(),
                    staging.buffer,
                    &[vk::BufferCopy {
                        src_offset: offset as u64,
                        dst_offset: staging.buffer_offset as u64,
                        size: layout.size() as u64,
                    }],
                );
            },
        );

        (
            status,
            Some((fun, staging.memory.as_ptr())),
            result.prev_access,
        )
    }
    unsafe fn write_buffer_impl(
        &mut self,
        buffer: &ObjRef<object::Buffer>,
        offset: usize,
        layout: std::alloc::Layout,

        cmd: vk::CommandBuffer,
        queue_family: u32,
        submission: QueueSubmission,

        check_availability: bool,
        device: &Device,

        fun: Box<dyn FnOnce(*mut u8) + Send + 'static>,
    ) -> (
        ResourceAccessStatus,
        Option<(Box<dyn FnOnce(*mut u8) + Send + 'static>, *mut u8)>,
        SmallVec<[QueueSubmission; 4]>,
    ) {
        if layout.size() == 0 {
            return (ResourceAccessStatus::None, None, SmallVec::new());
        }

        let allocator = device.allocator();
        let allocation = buffer.get_allocation();

        let allocation_info = allocator.get_allocation_info(allocation);

        let mut submissions = SmallVec::new();
        let mut status = ResourceAccessStatus::Recorded;
        let mut fun = Some(fun);

        let synchronization_result = buffer.access_mutable(
            |d| d.get_mutable_state(),
            |s| {
                // if the destination buffer is mapped, we'll write to it directly
                if let Ok(ptr) = allocator.map_memory(allocation) {
                    let ptr = ptr.add(offset);

                    assert!(
                        (ptr as usize & (layout.align() - 1)) == 0,
                        "Buffer allocation is not properly aligned"
                    );

                    let mapped_size = layout.size();
                    let mut fun = fun.take().unwrap();

                    let synchronization_state = &mut s.synchronization_state();

                    if check_availability {
                        synchronization_state.retain_active_submissions(device);
                    }

                    synchronization_state.update_host_access(|prev_access| {
                        if prev_access.is_empty() {
                            fun(ptr);

                            allocator.flush_allocation(
                                allocation,
                                offset as u64,
                                mapped_size as u64,
                            );

                            allocator.unmap_memory(allocation);

                            status = ResourceAccessStatus::Immediate;
                            HostAccessKind::Immediate
                        } else {
                            let (submission, semaphore) = device.make_submission(None);
                            let ptr = unsafe { UnsafeSend::new(ptr) };
                            let group = device.open_generation_finalized(move |device| {
                                fun(*ptr);

                                device.allocator().flush_allocation(
                                    allocation,
                                    offset as u64,
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

                            submissions.push(submission);

                            status = ResourceAccessStatus::HostDelayed;
                            HostAccessKind::Synchronized(submission)
                        }
                    });

                    None
                }
                // otherwise schedule a transfer operation
                else {
                    let result = s.synchronization_state().update(
                        queue_family,
                        TypeNone::new_none(),
                        &[submission],
                        TypeNone::new_none(),
                        queue_family,
                        buffer.get_create_info().sharing_mode_concurrent,
                    );

                    status = ResourceAccessStatus::Recorded;
                    Some(result)
                }
            },
        );

        let result = match status {
            ResourceAccessStatus::None => unreachable!(),
            ResourceAccessStatus::Immediate | ResourceAccessStatus::HostDelayed => {
                return (status, None, submissions);
            }
            ResourceAccessStatus::Recorded => {
                assert!(submissions.is_empty());
                synchronization_result.unwrap()
            }
        };

        assert!(result.transition_ownership_from.is_none(), "TODO");
        assert!(result.transition_layout_from.is_none(), "unreachable");

        let mut fun = fun.take().unwrap();

        let staging = self.ring.allocate(layout, submission, device);

        let d = device.device();
        debug_label_span(
            cmd,
            device,
            |f| {
                if let Some(label) = buffer.get_create_info().label.as_ref() {
                    write!(f, "Copy to '{label}'")
                } else {
                    write!(f, "Copy to buffer {:p}", buffer.get_handle())
                }
            },
            || {
                d.cmd_copy_buffer(
                    cmd,
                    staging.buffer,
                    buffer.get_handle(),
                    &[vk::BufferCopy {
                        dst_offset: staging.buffer_offset as u64,
                        src_offset: offset as u64,
                        size: layout.size() as u64,
                    }],
                );
            },
        );

        (
            status,
            Some((fun, staging.memory.as_ptr())),
            result.prev_access,
        )
    }
    unsafe fn read_image_impl(
        &mut self,
        image: &ObjRef<object::Image>,
        final_layout: Option<vk::ImageLayout>,
        texel_layout: std::alloc::Layout,
        regions: &[ImageRegion],

        cmd: vk::CommandBuffer,
        queue_family: u32,
        submission: QueueSubmission,

        check_availability: bool,

        device: &Device,
    ) -> (
        ResourceAccessStatus,
        SmallVec<[QueueSubmission; 4]>,
        SmallVec<[*const u8; 2]>,
    ) {
        if regions.is_empty() {
            return (ResourceAccessStatus::None, SmallVec::new(), SmallVec::new());
        }
        assert!(
            texel_layout.size() & (texel_layout.align() - 1) == 0,
            "Size is not a multiple of alignment"
        );

        let raw_image = image.get_handle();
        let allocator = device.allocator();
        let allocation = image.get_allocation();

        let result = image.access_mutable(
            |d| d.get_mutable_state(),
            |s| {
                let synchronization_state = s.synchronization_state();
                if check_availability {
                    synchronization_state.retain_active_submissions(device);
                }

                let result = synchronization_state.update(
                    queue_family,
                    TypeSome::new_some(vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                    &[submission],
                    TypeSome::new_some(
                        final_layout.unwrap_or(vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                    ),
                    queue_family,
                    image.get_create_info().sharing_mode_concurrent,
                );

                result
            },
        );

        assert!(result.transition_ownership_from.is_none(), "TODO");
        let d = device.device();

        let mut region_pointers = SmallVec::new();

        debug_label_span(
            cmd,
            device,
            |f| {
                if let Some(label) = image.get_create_info().label.as_ref() {
                    write!(f, "Copy to '{label}'")
                } else {
                    write!(f, "Copy to image {:p}", image.get_handle())
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
                        subresource_range: image.get_whole_subresource_range(),
                        ..Default::default()
                    };
                    let dependency_info = vk::DependencyInfoKHR {
                        image_memory_barrier_count: 1,
                        p_image_memory_barriers: &barrier,
                        ..Default::default()
                    };
                    d.cmd_pipeline_barrier_2_khr(cmd, &dependency_info);
                }

                self.emit_image_transfers(
                    texel_layout,
                    &regions,
                    submission,
                    device,
                    &mut |ptr, _, _| {
                        region_pointers.push(ptr as *const u8);
                    },
                    &mut |buffer, copies| {
                        d.cmd_copy_image_to_buffer(
                            cmd,
                            raw_image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            buffer,
                            copies,
                        )
                    },
                );
            },
        );

        (
            ResourceAccessStatus::Recorded,
            result.prev_access,
            region_pointers,
        )
    }
    unsafe fn write_image_impl(
        &mut self,
        image: &ObjRef<object::Image>,
        final_layout: Option<vk::ImageLayout>,
        do_transition: bool,
        texel_layout: std::alloc::Layout,
        regions: &[ImageRegion],

        cmd: vk::CommandBuffer,
        queue_family: u32,
        submission: QueueSubmission,

        check_availability: bool,

        device: &Device,

        fun: &mut dyn FnMut(*mut u8, usize, &ImageRegion),
    ) -> (ResourceAccessStatus, SmallVec<[QueueSubmission; 4]>) {
        if regions.is_empty() {
            return (ResourceAccessStatus::None, SmallVec::new());
        }
        assert!(
            texel_layout.size() & (texel_layout.align() - 1) == 0,
            "Size is not a multiple of alignment"
        );

        let raw_image = image.get_handle();
        let allocator = device.allocator();
        let allocation = image.get_allocation();

        let result = image.access_mutable(
            |d| d.get_mutable_state(),
            |s| {
                let synchronization_state = s.synchronization_state();
                if check_availability {
                    synchronization_state.retain_active_submissions(device);
                }

                let result = synchronization_state.update(
                    queue_family,
                    TypeSome::new_some(vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                    &[submission],
                    TypeSome::new_some(
                        final_layout.unwrap_or(vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                    ),
                    queue_family,
                    image.get_create_info().sharing_mode_concurrent,
                );

                result
            },
        );

        assert!(result.transition_ownership_from.is_none(), "TODO");
        let d = device.device();

        debug_label_span(
            cmd,
            device,
            |f| {
                if let Some(label) = image.get_create_info().label.as_ref() {
                    write!(f, "Copy to '{label}'")
                } else {
                    write!(f, "Copy to image {:p}", image.get_handle())
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
                        subresource_range: image.get_whole_subresource_range(),
                        ..Default::default()
                    };
                    let dependency_info = vk::DependencyInfoKHR {
                        image_memory_barrier_count: 1,
                        p_image_memory_barriers: &barrier,
                        ..Default::default()
                    };
                    d.cmd_pipeline_barrier_2_khr(cmd, &dependency_info);
                }

                self.emit_image_transfers(
                    texel_layout,
                    &regions,
                    submission,
                    device,
                    fun,
                    &mut |buffer, copies| {
                        d.cmd_copy_buffer_to_image(
                            cmd,
                            buffer,
                            raw_image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            copies,
                        )
                    },
                );

                if let Some(to) = final_layout {
                    if do_transition && to != vk::ImageLayout::TRANSFER_DST_OPTIMAL {
                        let barrier = vk::ImageMemoryBarrier2KHR {
                            src_stage_mask: vk::PipelineStageFlags2KHR::COPY,
                            src_access_mask: vk::AccessFlags2KHR::TRANSFER_WRITE,
                            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            new_layout: to,
                            image: raw_image,
                            subresource_range: image.get_whole_subresource_range(),
                            ..Default::default()
                        };
                        let dependency_info = vk::DependencyInfoKHR {
                            image_memory_barrier_count: 1,
                            p_image_memory_barriers: &barrier,
                            ..Default::default()
                        };
                        d.cmd_pipeline_barrier_2_khr(cmd, &dependency_info);
                    }
                }
            },
        );

        (ResourceAccessStatus::Recorded, result.prev_access)
    }
    unsafe fn emit_image_transfers(
        &mut self,
        texel_layout: std::alloc::Layout,
        regions: &[ImageRegion],
        submission: QueueSubmission,
        device: &Device,
        new_memory_fun: &mut dyn FnMut(*mut u8, usize, &ImageRegion),
        cmd_fun: &mut dyn FnMut(vk::Buffer, &[vk::BufferImageCopy]),
    ) {
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

            let staging = self.ring.allocate(layout, submission, device);
            new_memory_fun(staging.memory.as_ptr(), i, region);

            (staging.buffer, staging.buffer_offset, region)
        });

        let mut converted_regions: SmallVec<[vk::BufferImageCopy; 16]> = SmallVec::new();
        let mut last_buffer = None;
        let mut handle_region = |(buffer, offset, region): (vk::Buffer, usize, &ImageRegion)| {
            // if the source buffer changed (due to the last ringbuffer chunk getting filled)
            // we need to split the copy into multiple calls
            if let Some(last_buffer) = last_buffer {
                debug_assert!(last_buffer != vk::Buffer::null());
                if last_buffer != buffer {
                    debug_assert!(!converted_regions.is_empty());
                    cmd_fun(last_buffer, &converted_regions);
                    converted_regions.clear();
                }
            }

            if buffer != vk::Buffer::null() {
                converted_regions.push(vk::BufferImageCopy {
                    buffer_offset: offset as u64,
                    buffer_row_length: region.buffer_row_length,
                    buffer_image_height: region.buffer_image_height,
                    image_subresource: region.image_subresource.clone(),
                    image_offset: region.image_offset.clone(),
                    image_extent: region.image_extent.clone(),
                });
            }
            last_buffer = Some(buffer);
        };

        for r in region_sources {
            handle_region(r);
        }
        // flush the last buffer with a null buffer (specially handled)
        handle_region((vk::Buffer::null(), 0, regions.first().unwrap()));
    }
    pub unsafe fn flush(&mut self, device: &Device) -> Option<QueueSubmission> {
        self.command.flush(self.queue, device)
    }
    pub(crate) fn collect(&mut self, submission_manager: &SubmissionManager) {
        self.command.collect(submission_manager);
        self.ring.collect(submission_manager);
    }
    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
        self.command.destroy(device);
        self.ring.destroy(device);
    }
}

impl StagingManager {
    pub unsafe fn write_image<F: FnMut(*mut u8, usize, &ImageRegion) + Send + 'static>(
        &mut self,
        image: &ObjRef<object::Image>,
        final_layout: Option<vk::ImageLayout>,
        texel_layout: std::alloc::Layout,
        regions: Vec<ImageRegion>,
        device: &Device,
        mut fun: F,
    ) {
        let &mut OpenCommandBuffer {
            submission, buffer, ..
        } = self
            .command
            .get_current_command_buffer(self.queue, true, device);

        let queue_family = self.queue.family();

        let (status, wait) = self.write_image_impl(
            image,
            final_layout,
            true,
            texel_layout,
            &regions,
            buffer,
            queue_family,
            submission,
            true,
            device,
            &mut fun,
        );

        let OpenCommandBuffer {
            buffer,
            submission,
            wait_submissions,
            dirty,
            ..
        } = self
            .command
            .get_current_command_buffer(self.queue, true, device);

        wait_submissions.extend(wait);
        if status == ResourceAccessStatus::Recorded {
            *dirty = true;
        }
    }
    /// The caller must do the transition to final_layout and further synchronization on their own
    pub unsafe fn write_image_in_cmd<F: FnMut(*mut u8, usize, &ImageRegion) + Send + 'static>(
        &mut self,
        image: &ObjRef<object::Image>,
        final_layout: Option<vk::ImageLayout>,
        texel_layout: std::alloc::Layout,
        regions: Vec<ImageRegion>,

        cmd: vk::CommandBuffer,
        queue_family: u32,
        submission: QueueSubmission,

        device: &Device,

        mut fun: F,
    ) -> SmallVec<[QueueSubmission; 4]> {
        self.write_image_impl(
            image,
            final_layout,
            false,
            texel_layout,
            &regions,
            cmd,
            queue_family,
            submission,
            true,
            device,
            &mut fun,
        )
        .1
    }
    pub unsafe fn read_image(
        &mut self,
        image: &ObjRef<object::Image>,
        final_layout: Option<vk::ImageLayout>,
        texel_layout: std::alloc::Layout,
        regions: Vec<ImageRegion>,
        device: &Device,
        mut fun: impl FnMut(*const u8, usize, &ImageRegion) + Send + 'static,
    ) {
        let &mut OpenCommandBuffer {
            submission, buffer, ..
        } = self
            .command
            .get_current_command_buffer(self.queue, true, device);

        let queue_family = self.queue.family();

        let (status, wait, region_pointers) = self.read_image_impl(
            image,
            final_layout,
            texel_layout,
            &regions,
            buffer,
            queue_family,
            submission,
            true,
            device,
        );

        assert_eq!(status, ResourceAccessStatus::Recorded);
        assert_eq!(regions.len(), region_pointers.len());

        let OpenCommandBuffer {
            buffer,
            submission,
            wait_submissions,
            finalizers,
            dirty,
            ..
        } = self
            .command
            .get_current_command_buffer(self.queue, true, device);

        wait_submissions.extend(wait);
        if status == ResourceAccessStatus::Recorded {
            *dirty = true;
        }

        // override the vector to be Send, there will be no aliasing of the pointers
        let region_pointers = UnsafeSend::new(region_pointers);
        let finalizer = move |_: &Device| {
            for (i, (region, ptr)) in regions
                .iter()
                .zip(UnsafeSend::take(region_pointers))
                .enumerate()
            {
                fun(ptr, i, region);
            }
        };

        (*(finalizers.as_mut().unwrap().get())).push(Box::new(finalizer));
    }
    pub unsafe fn read_image_in_cmd(
        &mut self,
        image: &ObjRef<object::Image>,
        final_layout: Option<vk::ImageLayout>,
        texel_layout: std::alloc::Layout,
        regions: &[ImageRegion],

        cmd: vk::CommandBuffer,
        queue_family: u32,
        submission: QueueSubmission,

        device: &Device,
    ) -> (SmallVec<[QueueSubmission; 4]>, SmallVec<[*const u8; 2]>) {
        let (status, wait, region_pointers) = self.read_image_impl(
            image,
            final_layout,
            texel_layout,
            &regions,
            cmd,
            queue_family,
            submission,
            true,
            device,
        );

        assert_eq!(status, ResourceAccessStatus::Recorded);
        assert_eq!(regions.len(), region_pointers.len());

        (wait, region_pointers)
    }
    pub unsafe fn write_buffer(
        &mut self,
        buffer: &ObjRef<object::Buffer>,
        offset: usize,
        layout: std::alloc::Layout,
        immediate_fun: impl FnMut(*mut u8) + Send + 'static,

        device: &Device,
    ) {
        let &mut OpenCommandBuffer {
            submission,
            buffer: cmd,
            ..
        } = self
            .command
            .get_current_command_buffer(self.queue, true, device);

        let queue_family = self.queue.family();

        let (status, fun, wait) = self.write_buffer_impl(
            buffer,
            offset,
            layout,
            cmd,
            queue_family,
            submission,
            true,
            device,
            Box::new(immediate_fun),
        );

        let OpenCommandBuffer {
            buffer,
            submission,
            wait_submissions,
            dirty,
            finalizers,
            ..
        } = self
            .command
            .get_current_command_buffer(self.queue, true, device);

        wait_submissions.extend(wait);
        if status == ResourceAccessStatus::Recorded {
            *dirty = true;
        }

        if let Some((fun, ptr)) = fun {
            let ptr = UnsafeSend::new(ptr);
            let finalizer = move |_: &Device| fun(*ptr);

            // FIXME double allocation
            (*(finalizers.as_mut().unwrap().get())).push(Box::new(finalizer));
        }
    }
    pub unsafe fn read_buffer(
        &mut self,
        buffer: &ObjRef<object::Buffer>,
        offset: usize,
        layout: std::alloc::Layout,
        fun: impl FnMut(*const u8) + Send + 'static,

        device: &Device,
    ) {
        let &mut OpenCommandBuffer {
            submission,
            buffer: cmd,
            ..
        } = self
            .command
            .get_current_command_buffer(self.queue, true, device);

        let queue_family = self.queue.family();

        let (status, fun, wait) = self.read_buffer_impl(
            buffer,
            offset,
            layout,
            cmd,
            queue_family,
            submission,
            true,
            device,
            Box::new(fun),
        );

        let OpenCommandBuffer {
            buffer,
            submission,
            wait_submissions,
            dirty,
            finalizers,
            ..
        } = self
            .command
            .get_current_command_buffer(self.queue, true, device);

        wait_submissions.extend(wait);
        if status == ResourceAccessStatus::Recorded {
            *dirty = true;
        }

        if let Some((fun, ptr)) = fun {
            let ptr = UnsafeSend::new(ptr);
            let finalizer = move |_: &Device| fun(*ptr);

            // FIXME at this point the callback is contained in two boxes
            (*(finalizers.as_mut().unwrap().get())).push(Box::new(finalizer));
        }
    }
}

impl Device {
    pub unsafe fn write_buffer<F: FnMut(*mut u8) + Send + 'static>(
        &self,
        buffer: &ObjRef<object::Buffer>,
        offset: u64,
        layout: std::alloc::Layout,
        mut fun: F,
    ) -> QueueSubmission {
        todo!()
        // let mut write = self.staging_manager.write();
        // write.write_buffer(buffer, offset, layout, fun, self);
        // write.flush(None, self).unwrap()
    }
    pub unsafe fn write_image<F: FnMut(*mut u8, usize, &ImageRegion) + Send + 'static>(
        &self,
        image: &ObjRef<object::Image>,
        final_layout: Option<vk::ImageLayout>,
        texel_layout: std::alloc::Layout,
        regions: Vec<ImageRegion>,
        mut fun: F,
    ) -> QueueSubmission {
        let mut write = self.staging_manager.write();
        write.write_image(image, final_layout, texel_layout, regions, self, fun);
        write.flush(self).unwrap()
    }
    pub unsafe fn write_image_in_cmd<F: FnMut(*mut u8, usize, &ImageRegion) + Send + 'static>(
        &self,
        image: &ObjRef<object::Image>,
        final_layout: Option<vk::ImageLayout>,
        texel_layout: std::alloc::Layout,
        regions: Vec<ImageRegion>,
        cmd: vk::CommandBuffer,
        queue_family: u32,
        submission: QueueSubmission,
        mut fun: F,
    ) -> SmallVec<[QueueSubmission; 4]> {
        let mut write = self.staging_manager.write();
        write.write_image_in_cmd(
            image,
            final_layout,
            texel_layout,
            regions,
            cmd,
            queue_family,
            submission,
            self,
            fun,
        )
    }
    pub unsafe fn write_multiple<'a, R>(&self, fun: impl FnOnce(&mut StagingManager) -> R) -> R {
        let mut write = self.staging_manager.write();
        fun(&mut write)
    }
}

pub(crate) struct OpenCommandBuffer {
    wait_submissions: ahash::HashSet<QueueSubmission>,
    /// a list of callbacks to run when the command buffer execution finishes
    /// while not submitted, it may only be accessed by a mutable borrow of StagingManager
    /// after completion, the submission finalizer will mutably access it uncontended
    finalizers: Option<UnsafeSendSync<Arc<UnsafeCell<Vec<Box<dyn FnOnce(&Device) + Send>>>>>>,
    submission: QueueSubmission,
    semaphore: TimelineSemaphore,
    buffer: vk::CommandBuffer,
    dirty: bool,
}

enum CommandBufferState {
    Empty,
    Dirty(OpenCommandBuffer),
}

struct CommandBufferPool {
    label: Option<Cow<'static, str>>,
    pool: vk::CommandPool,
    free_buffers: Vec<vk::CommandBuffer>,
    current_buffer: Option<OpenCommandBuffer>,
    pending_buffers: VecDeque<(QueueSubmission, vk::CommandBuffer)>,
    // a monotonic counter used only for labeling command buffers
    next_buffer_index: u32,
}

impl CommandBufferPool {
    unsafe fn new(
        queue_family: u32,
        callbacks: Option<&vk::AllocationCallbacks>,
        label: Option<Cow<'static, str>>,
        device: &DeviceWrapper,
    ) -> Self {
        let info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::TRANSIENT
                | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: queue_family,
            ..Default::default()
        };

        Self {
            label,
            next_buffer_index: 0,
            pool: device.create_command_pool(&info, callbacks).unwrap(),
            free_buffers: Vec::new(),
            current_buffer: None,
            pending_buffers: VecDeque::new(),
        }
    }
    pub(crate) fn collect(&mut self, submission_manager: &SubmissionManager) {
        while let Some(&(submission, buffer)) = self.pending_buffers.front() {
            if submission_manager.is_submission_finished(submission) {
                self.pending_buffers.pop_front();
                self.free_buffers.push(buffer);
            } else {
                break;
            }
        }
    }
    unsafe fn get_current_command_buffer(
        &mut self,
        queue: submission::Queue,
        with_finalizers: bool,
        device: &Device,
    ) -> &mut OpenCommandBuffer {
        let mut current_buffer = self.current_buffer.take();
        current_buffer.get_or_insert_with(|| {
            let mut finalizers = None;
            let mut glue_finalizer: Option<Box<dyn FnOnce(&Device) + Send>> = None;

            if with_finalizers {
                // see OpenCommandBuffer.finalizers for safety regarding the UnsafeCell
                let arc: UnsafeSendSync<Arc<UnsafeCell<Vec<Box<dyn FnOnce(&Device) + Send>>>>> =
                    UnsafeSendSync::new(Arc::new(UnsafeCell::new(Vec::new())));

                let arc_copy = arc.clone();
                let finalizer = move |d: &Device| {
                    for finalizer in (*UnsafeCell::get(&arc_copy)).drain(..) {
                        finalizer(d);
                    }
                };

                finalizers = Some(arc);
                glue_finalizer = Some(Box::new(finalizer));
            }

            let (submission, semaphore) = device.make_submission(glue_finalizer);
            let buffer = match self.free_buffers.pop() {
                Some(b) => b,
                None => self.new_buffer(device),
            };

            device
                .device()
                .begin_command_buffer(
                    buffer,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap();

            OpenCommandBuffer {
                wait_submissions: constant_ahash_hashset(),
                finalizers,
                submission,
                semaphore,
                buffer,
                dirty: false,
            }
        });

        self.current_buffer = current_buffer;
        self.current_buffer.as_mut().unwrap()
    }
    unsafe fn new_buffer(&mut self, device: &Device) -> vk::CommandBuffer {
        let index = self.next_buffer_index;
        self.next_buffer_index += 1;

        let info = vk::CommandBufferAllocateInfo {
            command_pool: self.pool,
            level: vk::CommandBufferLevel::PRIMARY,
            // TODO allocate more buffers at a time?
            command_buffer_count: 1,
            ..Default::default()
        };

        let buffer = device.device().allocate_command_buffers(&info).unwrap()[0];
        let name = LazyDisplay(|f| {
            let label = self.label.as_deref().unwrap_or("CommandBufferPool");
            write!(f, "{label} buffer #{index}")
        });
        maybe_attach_debug_label(buffer, &name, device);

        buffer
    }
    unsafe fn flush(
        &mut self,
        queue: submission::Queue,
        device: &Device,
    ) -> Option<QueueSubmission> {
        if let Some(OpenCommandBuffer {
            wait_submissions,
            finalizers,
            submission,
            semaphore,
            buffer,
            dirty,
        }) = self.current_buffer.take()
        {
            self.pending_buffers.push_back((submission, buffer));

            let waits: SmallVec<[_; 16]> = {
                assert!(
                    !wait_submissions.contains(&submission),
                    "Cyclic dependency when submitting staging work"
                );

                let mut active: SmallVec<[_; 16]> = SmallVec::new();
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
