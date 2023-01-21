use std::{
    cell::UnsafeCell,
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

use ahash::HashSet;
use parking_lot::lock_api::{RawMutex, RawMutexTimed};
use pumice::VulkanResult;

use crate::{
    device::Device,
    storage::{constant_ahash_hashset, constant_ahash_randomstate},
};

use super::submission::{QueueSubmission, WaitResult};

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct GenerationId(u32);

struct OpenGenerationEntry {
    id: GenerationId,
    mutex: parking_lot::RawMutex,
    submissions: UnsafeCell<ahash::HashSet<QueueSubmission>>,
}

enum GenerationEntry {
    Open(Arc<OpenGenerationEntry>),
    Closed(Generation),
}

impl GenerationEntry {
    fn id(&self) -> GenerationId {
        match self {
            GenerationEntry::Open(arc) => arc.id,
            GenerationEntry::Closed(gen) => gen.id,
        }
    }
}

struct Generation {
    id: GenerationId,
    submissions: ahash::HashSet<QueueSubmission>,
}

impl Generation {
    fn new() -> Self {
        Self {
            id: GenerationId(0),
            submissions: HashSet::with_hasher(constant_ahash_randomstate()),
        }
    }
    fn next(prev: GenerationId) -> Self {
        Self {
            id: GenerationId(prev.0 + 1),
            submissions: HashSet::with_hasher(constant_ahash_randomstate()),
        }
    }
}

pub struct OpenGeneration(Arc<OpenGenerationEntry>);

impl OpenGeneration {
    pub fn id(&self) -> GenerationId {
        self.0.id
    }
    pub fn add_submissions(&self, submissions: impl IntoIterator<Item = QueueSubmission>) {
        // safe because as long as OpenGeneration exists, its lock is locked, and it cannot be cloned
        unsafe {
            self.0
                .submissions
                .get()
                .as_mut()
                .unwrap()
                .extend(submissions)
        };
    }
    pub fn finish(self) {
        drop(self)
    }
}

impl Drop for OpenGeneration {
    fn drop(&mut self) {
        unsafe { self.0.mutex.unlock() }
    }
}

/// A mechanism for a very coarse grained lifetime management
/// All submits are grouped into a generation and resource lifetimes can then be
/// checked against the generation that consumes them, in addition generations are
/// guaranteed to be a linear stream and to finish in order.
///
/// As such, lifetime tracking can be simplified so that it only keeps the most recent GenerationId
/// rather than a vector of all relevant queue submissions.
pub(crate) struct GenerationManager {
    max_in_flight: u32,
    next_id: GenerationId,
    batches: VecDeque<GenerationEntry>,
}

impl GenerationManager {
    pub(crate) fn new(max_in_flight: u32) -> Self {
        Self {
            max_in_flight,
            next_id: GenerationId(0),
            batches: VecDeque::new(),
        }
    }
    fn make_generation(&mut self) -> Generation {
        let generation = Generation {
            id: self.next_id,
            submissions: constant_ahash_hashset(),
        };
        self.next_id = GenerationId(self.next_id.0 + 1);
        generation
    }
    pub fn get_front_id(&self) -> Option<GenerationId> {
        self.batches.front().map(|a| a.id())
    }
    pub fn get_back_id(&self) -> Option<GenerationId> {
        self.batches.back().map(|a| a.id())
    }
    pub fn open_generation(&mut self, device: &Device) -> OpenGeneration {
        if self.batches.len() as u32 == self.max_in_flight {
            let id = self.get_front_id().unwrap();
            self.wait_for_generation_sequential(id, u64::MAX, device)
                .unwrap();
        }

        let generation = self.make_generation();
        let generation = Arc::new(OpenGenerationEntry {
            id: generation.id,
            mutex: parking_lot::RawMutex::INIT,
            submissions: UnsafeCell::new(constant_ahash_hashset()),
        });
        unsafe { generation.mutex.lock() };

        self.batches
            .push_back(GenerationEntry::Open(generation.clone()));

        OpenGeneration(generation)
    }
    fn inner_wait_for_generation(
        &mut self,
        index: usize,
        timeout_ns: u64,
        device: &Device,
    ) -> VulkanResult<WaitResult> {
        let next = &mut self.batches[index];

        let next = match next {
            GenerationEntry::Open(arc) => {
                if timeout_ns == 0 && arc.mutex.is_locked() {
                    return VulkanResult::Ok(WaitResult::Timeout);
                }

                if arc.mutex.try_lock_for(Duration::from_nanos(timeout_ns)) {
                    debug_assert!(Arc::strong_count(arc) == 1);
                } else {
                    return VulkanResult::Ok(WaitResult::Timeout);
                }

                let inner = unsafe { std::ptr::read(arc) };
                let OpenGenerationEntry {
                    id,
                    mutex: _,
                    submissions,
                } = Arc::try_unwrap(inner)
                    .ok()
                    .expect("Locking the mutex implies the OpenGeneration being dropped");

                unsafe {
                    std::ptr::write(
                        next,
                        GenerationEntry::Closed(Generation {
                            id,
                            submissions: UnsafeCell::into_inner(submissions),
                        }),
                    );
                }

                let GenerationEntry::Closed(gen) = next else {
                    unreachable!()
                };

                gen
            }
            GenerationEntry::Closed(gen) => gen,
        };

        let res =
            device.wait_for_submissions(next.submissions.iter().cloned(), timeout_ns, false)?;

        match res {
            WaitResult::Timeout => VulkanResult::Ok(WaitResult::Timeout),
            WaitResult::AllFinished => VulkanResult::Ok(WaitResult::AllFinished),
            WaitResult::AnyFinished => unreachable!(),
        }
    }
    pub fn wait_for_generation_single(
        &mut self,
        id: GenerationId,
        timeout_ns: u64,
        device: &Device,
    ) -> VulkanResult<WaitResult> {
        if let Ok(index) = self.batches.binary_search_by_key(&id, |b| b.id()) {
            match self.inner_wait_for_generation(index, timeout_ns, device)? {
                WaitResult::Timeout => return Ok(WaitResult::Timeout),
                WaitResult::AllFinished => {}
                WaitResult::AnyFinished => unreachable!(),
            }

            self.batches.remove(index);

            Ok(WaitResult::AllFinished)
        } else {
            Ok(WaitResult::AllFinished)
        }
    }
    pub fn wait_for_generation_sequential(
        &mut self,
        id: GenerationId,
        timeout_ns: u64,
        device: &Device,
    ) -> VulkanResult<WaitResult> {
        let start = Instant::now();
        let mut remaining_timeout_ns = timeout_ns;

        while let Some(next) = self.batches.front() {
            // batch is more recent than the one we are waiting for, we have nothing to do
            if next.id() > id {
                break;
            }

            match self.inner_wait_for_generation(0, remaining_timeout_ns, device)? {
                WaitResult::Timeout => return Ok(WaitResult::Timeout),
                WaitResult::AllFinished => {}
                WaitResult::AnyFinished => unreachable!(),
            }

            // u64::MAX is specially cased by the specification to never ever end, try to preserve it
            if timeout_ns != u64::MAX {
                remaining_timeout_ns = (timeout_ns as u128)
                    .saturating_sub(start.elapsed().as_nanos())
                    .try_into()
                    .unwrap();
            }

            self.batches.pop_front().unwrap();
        }

        VulkanResult::Ok(WaitResult::AllFinished)
    }
    pub(crate) unsafe fn clear(&mut self) {
        self.batches.clear();
    }
}

impl Device {
    pub fn open_generation(&self) -> OpenGeneration {
        self.generation_manager.write().open_generation(self)
    }
    /// Returns the id of the oldest generation that is possibly still unfinished,
    /// this is useful for bulk lifetime comparison, the finer grained alternative is `is_generation_finished`
    pub fn get_oldest_unfinished_generation(&self) -> GenerationId {
        let manager = self.generation_manager.read();
        manager.get_front_id().unwrap_or(manager.next_id)
    }
    /// Waits for only this generation to finish, generations added before this one are untouched.
    pub fn wait_for_generation_single(
        &self,
        id: GenerationId,
        timeout_ns: u64,
    ) -> VulkanResult<WaitResult> {
        self.generation_manager
            .write()
            .wait_for_generation_single(id, timeout_ns, self)
    }
    /// Waits for all preceding generations including this one to finish.
    pub fn wait_for_generation_sequential(
        &self,
        id: GenerationId,
        timeout_ns: u64,
    ) -> VulkanResult<WaitResult> {
        self.generation_manager
            .write()
            .wait_for_generation_sequential(id, timeout_ns, self)
    }
    /// Checks whether a generation has finished without issuing any vulkan commands,
    /// this relies on `wait_for_generation` being called sometime earlier
    pub fn is_generation_finished(&self, id: GenerationId) -> bool {
        self.get_oldest_unfinished_generation() > id
    }
}
