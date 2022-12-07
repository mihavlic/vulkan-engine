use std::{
    cmp::Ordering,
    collections::{BTreeSet, VecDeque},
    panic::AssertUnwindSafe,
};

use ahash::{HashSet, RandomState};
use pumice::{try_vk, util::result::VulkanResult};

use crate::{
    context::device::Device,
    submission::{QueueSubmission, WaitResult},
};

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct GenerationId(u32);

struct Generation {
    id: GenerationId,
    submissions: ahash::HashSet<QueueSubmission>,
}

impl Generation {
    fn new() -> Self {
        Self {
            id: GenerationId(0),
            submissions: HashSet::with_hasher(RandomState::new()),
        }
    }
    fn next(prev: GenerationId) -> Self {
        Self {
            id: GenerationId(prev.0 + 1),
            submissions: HashSet::with_hasher(RandomState::new()),
        }
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
    batches: VecDeque<Generation>,
}

impl GenerationManager {
    pub(crate) fn new(max_in_flight: u32) -> Self {
        Self {
            max_in_flight,
            batches: [Generation::new()].into(),
        }
    }
    pub fn add_submissions(&mut self, submissions: impl IntoIterator<Item = QueueSubmission>) {
        // batches is always non-empty
        let head = self.batches.back_mut().unwrap();
        head.submissions.extend(submissions);
    }
    pub fn end_generation(&mut self, device: &Device) -> VulkanResult<()> {
        // we currently have
        if self.batches.len() as u32 == self.max_in_flight {
            try_vk!(self.wait_for_generation(self.batches.front().unwrap().id, u64::MAX, device));
        }

        let head = self.batches.back_mut().unwrap();
        if !head.submissions.is_empty() {
            let id = head.id;
            self.batches.push_back(Generation::next(id));
        }

        VulkanResult::new_ok(())
    }
    pub fn get_current_generation(&self) -> GenerationId {
        self.batches.back().unwrap().id
    }
    pub fn get_oldest_unfinished_generation(&self) -> GenerationId {
        self.batches.front().unwrap().id
    }
    pub fn wait_for_generation(
        &mut self,
        id: GenerationId,
        timeout_ns: u64,
        device: &Device,
    ) -> VulkanResult<WaitResult> {
        while let Some(next) = self.batches.front_mut() {
            // batch is more recent than the one we are waiting for, we have nothing to do
            if next.id > id {
                break;
            }

            let res = try_vk!(device.wait_for_submissions(
                next.submissions.iter().cloned(),
                timeout_ns,
                false
            ));

            match res {
                WaitResult::Timeout => return VulkanResult::new_ok(WaitResult::Timeout),
                WaitResult::AllFinished => {}
                WaitResult::AnyFinished => unreachable!(),
            }

            let next = self.batches.pop_front().unwrap();
            // make sure that there's always one active generation
            if self.batches.is_empty() {
                self.batches.push_back(Generation::next(next.id));
            }
        }

        VulkanResult::new_ok(WaitResult::AllFinished)
    }
}

impl Device {
    pub fn add_submissions(&self, submissions: impl IntoIterator<Item = QueueSubmission>) {
        self.generation_manager.write().add_submissions(submissions)
    }
    pub fn end_generation(&self) -> VulkanResult<()> {
        self.generation_manager.write().end_generation(self)
    }
    pub fn get_current_generation(&self) -> GenerationId {
        self.generation_manager.read().get_current_generation()
    }
    /// Returns the id of the oldest generation that is possibly still unfinished,
    /// this is useful for bulk lifetime comparison, the single alternative is `is_generation_finished`
    pub fn get_oldest_unfinished_generation(&self) -> GenerationId {
        self.generation_manager
            .read()
            .get_oldest_unfinished_generation()
    }
    pub fn wait_for_generation(
        &self,
        id: GenerationId,
        timeout_ns: u64,
    ) -> VulkanResult<WaitResult> {
        self.generation_manager
            .write()
            .wait_for_generation(id, timeout_ns, self)
    }
    /// Checks whether a generation has finished without issuing any vulkan commands,
    /// this relies on `wait_for_generation` being called sometime earlier
    pub fn is_generation_finished(&self, id: GenerationId) -> bool {
        self.get_oldest_unfinished_generation() > id
    }
}
