use std::{
    mem::ManuallyDrop,
    time::{Duration, Instant},
};

use crate::{object, storage::constant_ahash_hashmap};

use super::{batch::GenerationId, submission::WaitResult, Device};

pub enum InflightResource {
    Image(object::Image),
    Buffer(object::Buffer),
    Closure(ManuallyDrop<Box<dyn FnOnce() + 'static>>),
}

impl Drop for InflightResource {
    fn drop(&mut self) {
        match self {
            InflightResource::Image(_) => {}
            InflightResource::Buffer(_) => {}
            InflightResource::Closure(closure) => unsafe {
                ManuallyDrop::take(closure)();
            },
        }
    }
}

pub struct InflightResourceManager {
    // TODO use a better data structure
    pending: Vec<(GenerationId, Vec<InflightResource>)>,
}

impl InflightResourceManager {
    pub(crate) fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }
    pub(crate) fn add<I: IntoIterator<Item = InflightResource>>(
        &mut self,
        id: GenerationId,
        iter: I,
    ) {
        match self.pending.binary_search_by_key(&id, |(k, _)| *k) {
            Ok(i) => self.pending[i].1.extend(iter),
            Err(i) => self.pending.insert(i, (id, iter.into_iter().collect())),
        }
    }
    pub(crate) fn poll(&mut self, ctx: &Device) {
        // TODO make this configurable, move to a thread
        let start = Instant::now();
        let mut i = 0;
        while i < self.pending.len() {
            let elapsed = start.elapsed().as_nanos();
            if elapsed > 2_000_000 {
                break;
            }
            let ns = (2_000_000 - elapsed).try_into().unwrap();

            let (id, _) = self.pending[i];
            match ctx.wait_for_generation_single(id, ns).unwrap() {
                WaitResult::Timeout => {
                    i += 1;
                }
                WaitResult::AllFinished => {
                    self.pending.remove(i);
                }
                WaitResult::AnyFinished => unreachable!(),
            }
        }
    }
}

impl Device {
    pub fn add_inflight<I: IntoIterator<Item = InflightResource>>(
        &self,
        generation: GenerationId,
        iter: I,
    ) {
        self.pending_resources.write().add(generation, iter);
    }
}
