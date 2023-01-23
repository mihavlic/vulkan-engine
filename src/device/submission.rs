use std::{
    ffi::c_void,
    sync::atomic::{AtomicBool, Ordering},
};

use pumice::{vk, VulkanResult};
use smallvec::SmallVec;

use crate::{
    arena::arena::{GenArena, U32Key},
    device::Device,
};

// little endian:
// [value bits, flag bit]
// big endian:
// [flag bit, value bits]
#[derive(Clone, Copy)]
struct SemaphoreValue(u64);

impl SemaphoreValue {
    fn new(value: u64, flag: bool) -> Self {
        assert!(value < 2u64.pow(63));

        #[cfg(target_endian = "little")]
        return Self(value << 1 | flag as u64);
        #[cfg(target_endian = "big")]
        return Self(value >> 1 | flag as u64);
    }
    fn value(&self) -> u64 {
        #[cfg(target_endian = "little")]
        return self.0 >> 1;
        #[cfg(target_endian = "big")]
        return self.0 << 1;
    }
    fn set_value(&mut self, value: u64) {
        assert!(value < 2u64.pow(63));

        self.0 &= 1;
        #[cfg(target_endian = "little")]
        let _ = self.0 |= value << 1;
        #[cfg(target_endian = "big")]
        let _ = self.0 |= value >> 1;
    }
    fn set_head(&mut self, flag: bool) {
        self.0 &= !1;
        self.0 |= flag as u64;
    }
    fn is_head(&self) -> bool {
        (self.0 & 1) == 1
    }
}

#[derive(Clone, Copy)]
pub struct Queue {
    pub(crate) raw: vk::Queue,
    pub(crate) family: u32,
}

impl Queue {
    pub(crate) fn new(raw: vk::Queue, family: u32) -> Self {
        Self { raw, family }
    }
    pub fn raw(&self) -> vk::Queue {
        self.raw
    }
    pub fn family(&self) -> u32 {
        self.family
    }
}

#[derive(Clone, Copy)]
struct SemaphoreEntry {
    raw: vk::Semaphore,
    value: SemaphoreValue,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimelineSemaphore {
    pub raw: vk::Semaphore,
    pub value: u64,
}

impl SemaphoreEntry {
    fn to_public(&self) -> TimelineSemaphore {
        TimelineSemaphore {
            raw: self.raw,
            value: self.value.value(),
        }
    }
    fn bump_value(&mut self) {
        self.value.set_value(self.value.value() + 1);
    }
}

pub struct QueueSubmitData {
    finished: AtomicBool,
    semaphore: SemaphoreEntry,
}

pub enum WaitResult {
    Timeout,
    AllFinished,
    AnyFinished,
}

pub(crate) struct SubmissionManager {
    submissions: GenArena<U32Key, QueueSubmitData>,
    free_semaphores: Vec<SemaphoreEntry>,
}

impl SubmissionManager {
    pub(crate) fn new() -> Self {
        Self {
            submissions: GenArena::new(),
            free_semaphores: Vec::new(),
        }
    }
    fn wait_for_submissions<T: IntoIterator<Item = QueueSubmission>>(
        &self,
        submissions: T,
        timeout_ns: u64,
        wait_any: bool,
        device: &Device,
    ) -> VulkanResult<WaitResult> {
        let mut remaining_timeout_ns = timeout_ns;
        let start = std::time::Instant::now();

        let mut timeout = false;

        // since we need to know which semaphores have finished, we must emulate the requested behaviour,
        // though wait_any is going to be iffy, the performance impact of doing all of this is unknown, however here:
        // https://gitlab.freedesktop.org/mesa/mesa/-/issues/6266
        // it is said that applications normally poll synchronization primitives so it should be reasonably efficient
        for s in submissions {
            let data = self.submissions.get(s.0);
            let finished = data.is_none() || data.unwrap().finished.load(Ordering::Relaxed) || {
                let data = data.unwrap();

                let semaphore = data.semaphore.raw;
                let value = data.semaphore.value.value();

                let info = vk::SemaphoreWaitInfoKHR {
                    flags: vk::SemaphoreWaitFlagsKHR::empty(),
                    semaphore_count: 1,
                    p_semaphores: &semaphore,
                    p_values: &value,
                    ..Default::default()
                };

                let result = unsafe {
                    device
                        .device()
                        .wait_semaphores_khr(&info, remaining_timeout_ns)?
                };

                match result {
                    vk::Result::SUCCESS => {
                        data.finished.store(true, Ordering::Relaxed);
                        true
                    }
                    vk::Result::TIMEOUT => {
                        timeout = true;
                        false
                    }
                    _ => unreachable!("wait_semaphores_khr cannot return other success codes"),
                }
            };

            if wait_any && finished {
                return VulkanResult::Ok(WaitResult::AnyFinished);
            }

            // u64::MAX is specially cased by the specification to never ever end, try to preserve it
            if timeout_ns != u64::MAX {
                remaining_timeout_ns = (timeout_ns as u128)
                    .saturating_sub(start.elapsed().as_nanos())
                    .try_into()
                    .unwrap();
            }
        }

        if timeout {
            return VulkanResult::Ok(WaitResult::Timeout);
        } else {
            return VulkanResult::Ok(WaitResult::AllFinished);
        }
    }
    fn allocate(&mut self, device: &Device) -> (QueueSubmission, TimelineSemaphore) {
        let mut semaphore = self.get_fresh_semaphore(device);
        semaphore.bump_value();
        (self.push_submission(semaphore, true), semaphore.to_public())
    }
    fn push_submission(&mut self, mut semaphore: SemaphoreEntry, head: bool) -> QueueSubmission {
        semaphore.value.set_head(head);
        let key = self.submissions.insert(QueueSubmitData {
            finished: AtomicBool::new(false),
            semaphore,
        });
        QueueSubmission(key)
    }
    fn get_fresh_semaphore(&mut self, device: &Device) -> SemaphoreEntry {
        let semaphore = self.free_semaphores.pop().unwrap_or_else(|| {
            let p_next = vk::SemaphoreTypeCreateInfoKHR {
                semaphore_type: vk::SemaphoreTypeKHR::TIMELINE,
                initial_value: 0,
                ..Default::default()
            };
            let info = vk::SemaphoreCreateInfo {
                p_next: &p_next as *const _ as *const c_void,
                ..Default::default()
            };
            let raw = unsafe {
                device
                    .device()
                    .create_semaphore(&info, device.allocator_callbacks())
                    .unwrap()
            };
            SemaphoreEntry {
                raw,
                value: SemaphoreValue::new(0, false),
            }
        });

        debug_assert!(semaphore.value.is_head() == false);

        semaphore
    }
    fn collect(&mut self) {
        let drain = self
            .submissions
            .drain_filter(|s| s.semaphore.value.is_head() && s.finished.load(Ordering::Relaxed))
            .map(|mut s| {
                s.semaphore.value.set_head(false);
                s.semaphore
            });
        self.free_semaphores.extend(drain);
    }
    /// Empty all pending submissions, this should be called after vkDeviceWaitIdle
    unsafe fn clear(&mut self) {
        self.submissions.clear()
    }
}

pub struct AllocateSequential<'a> {
    manager: parking_lot::RwLockWriteGuard<'a, SubmissionManager>,
    semaphore: SemaphoreEntry,
    last: Option<QueueSubmission>,
}

impl<'a> AllocateSequential<'a> {
    fn alloc(&mut self, _queue: Queue) -> (QueueSubmission, TimelineSemaphore) {
        self.semaphore.bump_value();
        let key = self.manager.push_submission(self.semaphore, false);
        self.last = Some(key);
        (key, self.semaphore.to_public())
    }
}

impl<'a> Drop for AllocateSequential<'a> {
    fn drop(&mut self) {
        if let Some(key) = self.last {
            self.manager
                .submissions
                .get_mut(key.0)
                .unwrap()
                .semaphore
                .value
                .set_head(true);
        } else {
            // no submission was allocated, return the semaphore
            self.manager.free_semaphores.push(self.semaphore.clone())
        }
    }
}

impl Device {
    pub fn wait_for_submissions(
        &self,
        submissions: impl IntoIterator<Item = QueueSubmission>,
        timeout_ns: u64,
        wait_any: bool,
    ) -> VulkanResult<WaitResult> {
        self.synchronization_manager.read().wait_for_submissions(
            submissions,
            timeout_ns,
            wait_any,
            self,
        )
    }
    pub fn allocate_sequential(&self) -> AllocateSequential<'_> {
        let mut this: parking_lot::RwLockWriteGuard<'_, _> = self.synchronization_manager.write();
        AllocateSequential {
            semaphore: this.get_fresh_semaphore(self),
            manager: this,
            last: None,
        }
    }
    pub fn make_submission(&self) -> (QueueSubmission, TimelineSemaphore) {
        self.synchronization_manager.write().allocate(self)
    }
    pub fn idle_cleanup_poll(&self) {
        self.synchronization_manager.write().collect();
        self.pending_resources.write().poll(self);
    }
    pub fn wait_idle(&self) {
        // let mut submissions = self.synchronization_manager.write();
        // let mut generations = self.generation_manager.write();

        unsafe {
            self.device().device_wait_idle().unwrap();
            // it seems that device_wait_idle doesn't make validation layers consider all command buffers as having completed
            // so not waiting on them while destroying their command buffer is an error
            // submissions.clear();
            // generations.clear();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueueSubmission(U32Key);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReaderWriterState {
    Read(SmallVec<[QueueSubmission; 4]>),
    Write(QueueSubmission),
    None,
}

impl ReaderWriterState {
    pub fn write(&mut self) -> Synchronize {
        todo!()
    }
    pub fn read(&mut self) -> Synchronize {
        todo!()
    }
}

pub enum Synchronize {
    Barrier {},
    None,
}