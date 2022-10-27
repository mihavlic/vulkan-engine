use std::{
    borrow::Borrow,
    ffi::c_void,
    sync::atomic::{AtomicBool, Ordering},
};

use pumice::{try_vk, util::result::VulkanResult, vk, DeviceWrapper};
use smallvec::SmallVec;

use crate::{
    arena::{
        arena::{GenArena, U32Key},
        optional::OptionalU32,
    },
    context::device::Device,
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
    fn set_flag(&mut self, flag: bool) {
        self.0 &= !1;
        self.0 |= flag as u64;
    }
    fn flag(&self) -> bool {
        (self.0 & 1) == 1
    }
}

#[derive(Clone)]
pub struct Queue {
    raw: vk::Queue,
    family: u32,
}

#[derive(Clone, Copy)]
pub struct Semaphore {
    raw: vk::Semaphore,
    value: SemaphoreValue,
}

impl Semaphore {
    fn bump_value(&mut self) {
        self.value.set_value(self.value.value() + 1);
    }
}

pub struct QueueSubmitData {
    finished: AtomicBool,
    queue: Queue,
    semaphore: Semaphore,
}

pub enum WaitResult {
    Timeout,
    AllFinished,
    SingleFinished,
}

pub(crate) struct InnerSynchronizationManager {
    submissions: GenArena<U32Key, QueueSubmitData>,
    free_semaphores: Vec<Semaphore>,
}

impl InnerSynchronizationManager {
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
        let mut timeout_ns = timeout_ns;
        let start = std::time::Instant::now();

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

                let result = unsafe { device.device().wait_semaphores_khr(&info, timeout_ns) };

                match result.raw {
                    vk::Result::SUCCESS => {
                        data.finished.store(true, Ordering::Relaxed);
                        true
                    }
                    vk::Result::TIMEOUT => {
                        return VulkanResult::new_ok(WaitResult::Timeout);
                    }
                    _ => return VulkanResult::new_err(result.raw),
                }
            };

            if wait_any && finished {
                return VulkanResult::new_ok(WaitResult::SingleFinished);
            }

            if timeout_ns > 0 {
                let elapsed = start.elapsed().as_nanos();
                let timeout = timeout_ns as u128;
                if elapsed >= timeout {
                    return VulkanResult::new_ok(WaitResult::Timeout);
                } else {
                    timeout_ns = (timeout - elapsed) as u64;
                }
            }
        }

        return VulkanResult::new_ok(WaitResult::AllFinished);
    }
    fn allocate(&mut self, queue: Queue, device: &Device) -> QueueSubmission {
        let semaphore = self.get_fresh_semaphore(device);
        self.push_submission(queue, semaphore, true)
    }
    fn push_submission(
        &mut self,
        queue: Queue,
        mut semaphore: Semaphore,
        head: bool,
    ) -> QueueSubmission {
        semaphore.value.set_flag(head);
        let key = self.submissions.insert(QueueSubmitData {
            finished: AtomicBool::new(false),
            queue,
            semaphore,
        });
        QueueSubmission(key)
    }
    fn get_fresh_semaphore(&mut self, device: &Device) -> Semaphore {
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
            Semaphore {
                raw,
                value: SemaphoreValue::new(0, false),
            }
        });

        debug_assert!(semaphore.value.flag() == false);

        semaphore
    }
    fn collect(&mut self) {
        let drain = self
            .submissions
            .drain_filter(|s| s.semaphore.value.flag() && s.finished.load(Ordering::Relaxed))
            .map(|s| s.semaphore);
        self.free_semaphores.extend(drain);
    }
}

pub struct AllocateSequential<'a> {
    manager: parking_lot::RwLockWriteGuard<'a, InnerSynchronizationManager>,
    semaphore: Semaphore,
    last: Option<QueueSubmission>,
}

impl<'a> AllocateSequential<'a> {
    fn alloc(&mut self, queue: Queue) -> QueueSubmission {
        self.semaphore.bump_value();
        let key = self.manager.push_submission(queue, self.semaphore, false);
        self.last = Some(key);
        key
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
                .set_flag(true);
        } else {
            // no submission was allocated, return the semaphore
            self.manager.free_semaphores.push(self.semaphore.clone())
        }
    }
}

impl Device {
    pub fn wait_for_submissions<T: IntoIterator<Item = QueueSubmission>>(
        &self,
        submissions: T,
        timeout_ns: u64,
        wait_any: bool,
    ) -> VulkanResult<WaitResult> {
        self.0.synchronization_manager.read().wait_for_submissions(
            submissions,
            timeout_ns,
            wait_any,
            self,
        )
    }
    pub fn allocate_sequential(&self) -> AllocateSequential<'_> {
        let mut this: parking_lot::RwLockWriteGuard<'_, _> = self.0.synchronization_manager.write();
        AllocateSequential {
            semaphore: this.get_fresh_semaphore(self),
            manager: this,
            last: None,
        }
    }
    pub fn allocate(&self, queue: Queue) -> QueueSubmission {
        self.0.synchronization_manager.write().allocate(queue, self)
    }
    fn collect(&mut self) {
        self.0.synchronization_manager.write().collect();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
