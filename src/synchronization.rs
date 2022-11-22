use std::{
    borrow::Borrow,
    ffi::c_void,
    sync::atomic::{AtomicBool, Ordering},
};

use pumice::{try_vk, util::result::VulkanResult, vk, DeviceWrapper};
use smallvec::SmallVec;

use crate::{
    arena::{
        arena::{GenArena, Key, U32Key},
        uint::OptionalU32,
    },
    context::device::Device,
};

#[derive(Clone)]
pub struct Queue {
    raw: vk::Queue,
    family: u32,
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
pub struct Semaphore {
    raw: vk::Semaphore,
    value: u64,
}

impl Semaphore {
    fn bump_value(&mut self) {
        self.value += 1;
    }
}

pub struct QueueSubmitData {
    finished: AtomicBool,
    is_head: bool,
    prev: Option<QueueSubmission>,
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
                let value = data.semaphore.value;

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
                        let mut data = data;
                        // we traverse all the previous values and mark them as finished
                        loop {
                            let prev = data.finished.fetch_and(true, Ordering::Relaxed);
                            // the submission has already been waited on and its prev chain has been updated
                            if prev == true {
                                break;
                            }
                            // visit all the previous values of this semaphore, since this one is finished
                            // all the previous ones must be too, we update them to prevent
                            if let Some(key) = data.prev {
                                data = self.submissions.get(key.0).unwrap();
                            } else {
                                break;
                            }
                        }

                        true
                    }
                    vk::Result::TIMEOUT => {
                        timeout = true;
                        false
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
                    // we still want to check the other semaphores' status
                    timeout_ns = 0;
                } else {
                    timeout_ns = (timeout - elapsed) as u64;
                }
            }
        }

        if timeout {
            return VulkanResult::new_ok(WaitResult::Timeout);
        } else {
            return VulkanResult::new_ok(WaitResult::AllFinished);
        }
    }
    fn allocate(&mut self, queue: Queue, device: &Device) -> QueueSubmission {
        let mut semaphore = self.get_free_semaphore(device);
        semaphore.bump_value();
        self.push_submission(queue, semaphore, true, None)
    }
    fn push_submission(
        &mut self,
        queue: Queue,
        semaphore: Semaphore,
        is_head: bool,
        prev: Option<QueueSubmission>,
    ) -> QueueSubmission {
        let key = self.submissions.insert(QueueSubmitData {
            finished: AtomicBool::new(false),
            is_head,
            prev,
            queue,
            semaphore,
        });
        QueueSubmission(key)
    }
    /// Returns a new semaphore that is free to use, its value field is its current value
    /// to use it in a wait operation, you need to call bump_value()
    fn get_free_semaphore(&mut self, device: &Device) -> Semaphore {
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
            Semaphore { raw, value: 0 }
        });

        semaphore
    }
    fn collect(&mut self) {
        let drain = self
            .submissions
            .drain_filter(|s| {
                if s.finished.load(Ordering::Relaxed) {
                    // we need to make sure to only return the finished heads, because otherwise there will be other timeline values still in submission
                    if s.is_head {
                        self.free_semaphores.push(s.semaphore);
                    }
                    true
                } else {
                    false
                }
            })
            .map(|s| s.semaphore);
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
        let key = self
            .manager
            .push_submission(queue, self.semaphore, false, self.last);
        self.last = Some(key);
        key
    }
}

impl<'a> Drop for AllocateSequential<'a> {
    fn drop(&mut self) {
        if let Some(key) = self.last {
            self.manager.submissions.get_mut(key.0).unwrap().is_head = true;
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
            semaphore: this.get_free_semaphore(self),
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
