use std::{
    ffi::c_void,
    mem::ManuallyDrop,
    ops::Not,
    sync::atomic::{AtomicBool, Ordering},
};

use parking_lot::{MappedRwLockReadGuard, RwLockReadGuard, RwLockWriteGuard};
use pumice::{vk, VulkanResult};
use smallvec::SmallVec;

use crate::{
    arena::arena::{GenArena, U32Key},
    device::{
        debug::{maybe_attach_debug_label, DisplayConcat},
        Device,
    },
    graph::{
        compile::{
            BufferBarrier, CombinedResourceHandle, GraphResource, ImageBarrier,
            ResourceFirstAccessInterface,
        },
        execute::{DummySubmission, SubmissionExtra},
        resource_marker::{ResourceMarker, TypeOption},
        task::SendUnsafeCell,
    },
    object::{SynchronizationState, SynchronizeResult},
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

pub struct AtomicOption<T: Send> {
    empty: AtomicBool,
    data: SendUnsafeCell<ManuallyDrop<T>>,
}

// https://users.rust-lang.org/t/atomic-option-like-take/33880
impl<T: Send> AtomicOption<T> {
    pub fn new(data: T) -> Self {
        Self {
            empty: AtomicBool::new(false),
            data: SendUnsafeCell::new(ManuallyDrop::new(data)),
        }
    }
    pub fn take(&self) -> Option<T> {
        let empty = self.empty.swap(true, Ordering::Relaxed);
        if empty {
            return None;
        } else {
            let data = unsafe { ManuallyDrop::take(self.data.get_mut()) };
            return Some(data);
        }
    }
    pub fn is_none(&self) -> bool {
        self.empty.load(Ordering::Relaxed)
    }
    pub fn is_some(&self) -> bool {
        !self.empty.load(Ordering::Relaxed)
    }
}

impl<T: Send> Drop for AtomicOption<T> {
    fn drop(&mut self) {
        let _ = self.take();
    }
}

unsafe impl<T: Send> Send for AtomicOption<T> {}

struct SubmissionEntry {
    finalizer: AtomicOption<Option<Box<dyn FnOnce(&Device) + Send>>>,
    semaphore: SemaphoreEntry,
}

impl SubmissionEntry {
    fn to_public(&self) -> TimelineSemaphore {
        self.semaphore.to_public()
    }
    fn is_finished(&self) -> bool {
        self.finalizer.is_none()
    }
    fn mark_finished(&self, device: &Device) {
        if let Some(finalizer) = self.finalizer.take().unwrap() {
            finalizer(device);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WaitResult {
    Timeout,
    AllFinished,
    AnyFinished,
}

pub(crate) struct SubmissionManager {
    submissions: GenArena<U32Key, SubmissionEntry>,
    free_semaphores: Vec<SemaphoreEntry>,
    semaphore_count: u32,
}

impl SubmissionManager {
    pub(crate) fn new() -> Self {
        Self {
            submissions: GenArena::new(),
            free_semaphores: Vec::new(),
            semaphore_count: 0,
        }
    }
    pub(crate) fn wait_for_submissions<T: IntoIterator<Item = QueueSubmission>>(
        &self,
        submissions: T,
        timeout_ns: u64,
        wait_any: bool,
        device: &Device,
    ) -> VulkanResult<WaitResult> {
        let mut remaining_timeout_ns = timeout_ns;
        // avoid the syscall(?) when the timeout is infinite
        let skip_time = timeout_ns == 0 || timeout_ns == u64::MAX;
        let start = skip_time.not().then(|| std::time::Instant::now());

        let mut timeout = false;

        // since we need to know which semaphores have finished, we must emulate the requested behaviour,
        // though wait_any is going to be iffy, the performance impact of doing all of this is unknown, however here:
        // https://gitlab.freedesktop.org/mesa/mesa/-/issues/6266
        // it is said that applications normally poll synchronization primitives so it should be reasonably efficient
        for s in submissions {
            let data = self.submissions.get(s.0);
            let finished = data.is_none() || data.unwrap().is_finished() || {
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
                        data.mark_finished(device);
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
            if !skip_time {
                remaining_timeout_ns = (timeout_ns as u128)
                    .saturating_sub(start.unwrap().elapsed().as_nanos())
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
    pub(crate) fn allocate(
        &mut self,
        finalizer: Option<Box<dyn FnOnce(&Device) + Send>>,
        device: &Device,
    ) -> (QueueSubmission, TimelineSemaphore) {
        let mut semaphore = self.get_fresh_semaphore(device);
        semaphore.bump_value();
        (
            self.push_submission(semaphore, finalizer, true),
            semaphore.to_public(),
        )
    }
    fn push_submission(
        &mut self,
        semaphore: SemaphoreEntry,
        finalizer: Option<Box<dyn FnOnce(&Device) + Send>>,
        head: bool,
    ) -> QueueSubmission {
        let mut semaphore = semaphore;
        semaphore.value.set_head(head);

        let key = self.submissions.insert(SubmissionEntry {
            finalizer: AtomicOption::new(finalizer),
            semaphore,
        });
        QueueSubmission(key)
    }
    fn get_fresh_semaphore(&mut self, device: &Device) -> SemaphoreEntry {
        let semaphore = self.free_semaphores.pop().unwrap_or_else(|| {
            let raw = unsafe { device.create_raw_timeline_semaphore().unwrap() };
            maybe_attach_debug_label(
                raw,
                &DisplayConcat::new(&[&"Submission timeline semaphore ", &self.semaphore_count]),
                device,
            );
            self.semaphore_count += 1;
            SemaphoreEntry {
                raw,
                value: SemaphoreValue::new(0, false),
            }
        });

        debug_assert!(semaphore.value.is_head() == false);

        semaphore
    }
    pub(crate) fn is_submission_finished(&self, submission: QueueSubmission) -> bool {
        self.submissions.get(submission.0).is_none()
    }
    pub(crate) fn get_submission_data(
        &self,
        submission: QueueSubmission,
    ) -> Option<TimelineSemaphore> {
        self.submissions
            .get(submission.0)
            .map(SubmissionEntry::to_public)
    }
    fn collect(&mut self, device: &Device) {
        for (k, s) in self.submissions.iter().rev() {
            if !s.is_finished() {
                self.wait_for_submissions(std::iter::once(QueueSubmission(k)), 0, false, device)
                    .unwrap();
            }
        }

        let drain = self
            .submissions
            .drain_filter(|s| s.semaphore.value.is_head() && s.is_finished())
            .map(|mut s| {
                s.semaphore.value.set_head(false);
                s.semaphore
            });
        self.free_semaphores.extend(drain);
    }
    /// Empty all pending submissions, this should be called after vkDeviceWaitIdle
    unsafe fn clear(&mut self) {
        self.submissions.clear();
    }
    /// This waits on all currently active semaphores, essentially a vkDeviceWaitIdle that doesn't bother validation layers
    pub(crate) fn wait_all(&mut self, device: &Device) -> VulkanResult<()> {
        for (k, s) in self.submissions.iter().rev() {
            if s.semaphore.value.is_head() {
                self.wait_for_submissions(
                    std::iter::once(QueueSubmission(k)),
                    u64::MAX,
                    false,
                    device,
                )?;
                let mut semaphore = s.semaphore;
                semaphore.value.set_head(false);
                self.free_semaphores.push(semaphore);
            }
        }

        self.submissions.clear();
        Ok(())
    }
    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
        for semaphore in self.free_semaphores.drain(..) {
            device
                .device()
                .destroy_semaphore(semaphore.raw, device.allocator_callbacks());
        }
    }
}

pub struct AllocateSequential<'a> {
    manager: parking_lot::RwLockWriteGuard<'a, SubmissionManager>,
    semaphore: SemaphoreEntry,
    last: Option<QueueSubmission>,
}

impl<'a> AllocateSequential<'a> {
    fn alloc(
        &mut self,
        finalizer: Option<Box<dyn FnOnce(&Device) + Send>>,
    ) -> (QueueSubmission, TimelineSemaphore) {
        self.semaphore.bump_value();
        let key = self
            .manager
            .push_submission(self.semaphore, finalizer, false);
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
    pub fn make_submission(
        &self,
        finalizer: Option<Box<dyn FnOnce(&Device) + Send>>,
    ) -> (QueueSubmission, TimelineSemaphore) {
        self.synchronization_manager
            .write()
            .allocate(finalizer, self)
    }
    pub fn is_submission_finished(&self, submission: QueueSubmission) -> bool {
        self.synchronization_manager
            .read()
            .is_submission_finished(submission)
    }
    pub fn retain_active_submissions<A: smallvec::Array<Item = QueueSubmission>>(
        &self,
        submissions: &mut SmallVec<A>,
    ) {
        let manager = self.synchronization_manager.read();
        submissions.retain(|s| !manager.is_submission_finished(*s));
    }
    pub fn get_submission_data(&self, submission: QueueSubmission) -> Option<TimelineSemaphore> {
        self.synchronization_manager
            .read()
            .get_submission_data(submission)
    }
    pub fn collect_active_submission_datas(
        &self,
        submissions: impl IntoIterator<Item = QueueSubmission>,
        mut extend: &mut impl Extend<TimelineSemaphore>,
    ) {
        let guard = self.synchronization_manager.read();
        let iter = submissions.into_iter().filter_map(|s| {
            guard
                .submissions
                .get(s.0)
                .filter(|e| !e.is_finished())
                .map(SubmissionEntry::to_public)
        });
        extend.extend(iter);
    }
    pub fn filter_active_submissions<E: Extend<QueueSubmission>>(
        &self,
        submissions: impl IntoIterator<Item = QueueSubmission>,
        mut extend: &mut E,
    ) {
        let guard = self.synchronization_manager.read();
        let iter = submissions.into_iter().filter_map(|s| {
            guard
                .submissions
                .get(s.0)
                .filter(|e| !e.is_finished())
                .map(|_| s)
        });
        extend.extend(iter);
    }
    pub fn idle_cleanup_poll(&self) {
        let mut submission = self.synchronization_manager.write();
        submission.collect(self);
        let submission = RwLockWriteGuard::downgrade(submission);
        self.generation_manager.write().collect(&submission, self);
        self.staging_manager.write().collect(&submission);
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
    /// This achieves the idle state by waiting for all registered semaphores to finish.
    /// It likely has large overhead, however the validation layers do not consider device_wait_idle
    /// to make all command buffer not pending so it throws warnings during teardown.
    pub fn wait_idle_precise(&self) {
        self.synchronization_manager.write().wait_all(self);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueueSubmission(U32Key);
