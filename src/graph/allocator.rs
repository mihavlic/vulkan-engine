use super::{GraphResource, SubmissionResourceReuse};
use crate::{device::Device, simple_handle, storage::constant_ahash_hashset};
use pumice::{vk, VulkanResult};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    collections::{BTreeMap, BTreeSet},
    hash::Hash,
    rc::Rc,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryKind {
    Linear,
    Optimal,
    None,
}

impl From<vk::ImageTiling> for MemoryKind {
    fn from(value: vk::ImageTiling) -> Self {
        match value {
            vk::ImageTiling::OPTIMAL => Self::Optimal,
            vk::ImageTiling::LINEAR => Self::Linear,
            _ => unreachable!(),
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub(crate) struct AvailabilityToken(u32);

impl AvailabilityToken {
    pub(crate) const NONE: Self = Self(0);
    pub(crate) fn new() -> Self {
        Self(1)
    }
    pub(crate) fn bump(&mut self) {
        self.0 += 1;
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct MemorySuballocationKey {
    availability_token: AvailabilityToken,
    size: u64,
    id: u32,
}

pub(crate) struct MemorySuballocation {
    pub(crate) start: u64,
    // redundant copy, kinda ugly to fix
    pub(crate) key: MemorySuballocationKey,
    pub(crate) start_padding: u64,
    pub(crate) kind: MemoryKind,
    pub(crate) prev: Option<Rc<RefCell<MemorySuballocation>>>,
    pub(crate) next: Option<Rc<RefCell<MemorySuballocation>>>,
}

pub(crate) struct MemoryBlock {
    pub(crate) allocation: pumice_vma::Allocation,
    pub(crate) size: u64,
    pub(crate) align: u64,
    pub(crate) memory_type: u32,
    pub(crate) head: Rc<RefCell<MemorySuballocation>>,
}

pub(crate) type Suballocation = Rc<RefCell<MemorySuballocation>>;

pub(crate) struct SuballocationUgh {
    pub(crate) memory: Rc<MemoryBlock>,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

pub(crate) struct RcPtrComparator<T>(pub(crate) Rc<T>);

impl<T> PartialEq for RcPtrComparator<T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(Rc::as_ptr(&self.0), Rc::as_ptr(&other.0))
    }
}
impl<T> Eq for RcPtrComparator<T> {}
impl<T> Hash for RcPtrComparator<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Rc::as_ptr(&self.0), state);
    }
}

pub(crate) struct Suballocator {
    allocations:
        BTreeMap<MemorySuballocationKey, (Rc<RefCell<MemorySuballocation>>, Rc<MemoryBlock>)>,
    id_counter: u32,
}

impl Suballocator {
    pub(crate) fn new() -> Self {
        Self {
            allocations: BTreeMap::new(),
            id_counter: 0,
        }
    }
    pub(crate) fn reset(&mut self) {
        let memory_blocks = self.collect_blocks();

        self.allocations.clear();
        self.id_counter = 0;

        for RcPtrComparator(block) in memory_blocks {
            let key = MemorySuballocationKey {
                availability_token: AvailabilityToken::NONE,
                size: block.size,
                id: self.make_new_id(),
            };

            let mut head = block.head.borrow_mut();
            assert!(head.prev.is_none());

            head.start = 0;
            head.key = key;
            head.start_padding = 0;
            head.kind = MemoryKind::None;
            head.prev = None;
            head.next = None;

            drop(head);

            self.allocations.insert(key, (block.head.clone(), block));
        }
    }
    pub(crate) fn collect_blocks(&self) -> ahash::HashSet<RcPtrComparator<MemoryBlock>> {
        let mut memory_blocks = constant_ahash_hashset();
        for (key, (alloc, memory)) in &self.allocations {
            memory_blocks.insert(RcPtrComparator(memory.clone()));
        }

        memory_blocks
    }
    // assigns physical memory to a resource, it is not usually neccessary to free it since lifetime ranges are used for aliasing
    pub(crate) fn allocate<
        F: FnOnce() -> VulkanResult<(pumice_vma::Allocation, pumice_vma::AllocationInfo)>,
    >(
        &mut self,
        size: u64,
        align: u64,
        kind: MemoryKind,
        new_availability: AvailabilityToken,
        image_buffer_granularity: u64,
        memory_type_index: u32,
        reuse: &SubmissionResourceReuse,
        alloc_fun: F,
    ) -> VulkanResult<SuballocationUgh> {
        assert!(kind != MemoryKind::None);
        assert!(size > 0);
        assert!(align > 0);

        // TODO if we don't find an allocation after some maximum number of attempts, we allocate fresh memory
        for &valid in &reuse.current_available_intervals {
            let start = MemorySuballocationKey {
                availability_token: valid,
                size,
                id: 0,
            };
            for (key, (chunk, memory)) in self.allocations.range(start..) {
                // we could do this with a bounded range but this a bit is shorter
                if key.availability_token != valid {
                    break;
                }

                let chunk = chunk.borrow();
                assert!(*key == chunk.key);

                let mut alignment = align;
                if let Some(prev) = chunk.prev.as_ref() {
                    if kind != MemoryKind::None && kind != prev.borrow().kind {
                        alignment = alignment.max(image_buffer_granularity);
                    }
                }

                // we add the align because otherwise start may be 0 (which is only an offset into the block) and would mess up rounding
                let useful_start = round_up_pow2(chunk.start + memory.align, align) - memory.align;
                let end = chunk.start + key.size;
                let mut useful_end = end;

                if let Some(prev) = chunk.next.as_ref() {
                    if kind != MemoryKind::None && kind != prev.borrow().kind {
                        useful_end = round_down_pow2(end, image_buffer_granularity);
                    }
                }

                // round has put us past the end of the allocation
                if useful_start >= useful_end {
                    continue;
                }

                // allocation can't fit in this chunk
                if useful_end - useful_start < size {
                    continue;
                }

                // at this point we know we can make the allocation
                // we will store the left offset required to reach alignment in the suballocation itself, as the offset is always? smaller than any other possible allocation
                // on the right, we may create a suballocation with the previous availability containing the remainder of the previous chunk

                // |---------------chunk---------------|
                // | start pad | new chunk | remainder |
                //             ^ useful_start

                let alloc_end = useful_start + size;
                let start_pad = useful_start - chunk.start;
                let remainder = end - alloc_end;

                const REMAINDER_MIN_SIZE: u64 = 256;

                // borrowchecker be like
                let key = {
                    let new_key = *key;
                    drop(key);
                    drop(chunk);
                    new_key
                };

                let (mut chunk_value, memory) = self
                    .allocations
                    .remove(&key)
                    .expect("The chunk is not here?!");
                let mut chunk = chunk_value.borrow_mut();

                chunk.start_padding = start_pad;

                // TODO TODO merge neighboring chunks that share a live range

                // if the remainder is big enough, it is made into a separate chunk
                // ensure that empty chunks sre at least as large as image_buffer_granularity
                if remainder < REMAINDER_MIN_SIZE.max(image_buffer_granularity) {
                    let none = self.allocations.insert(
                        MemorySuballocationKey {
                            availability_token: new_availability,
                            ..key
                        },
                        (chunk_value.clone(), memory.clone()),
                    );
                    assert!(none.is_none());
                } else {
                    let id = self.make_new_id();
                    let remainder_key = MemorySuballocationKey {
                        // copy the old availability
                        availability_token: key.availability_token,
                        size: remainder,
                        id,
                    };
                    let next = chunk.next.take();
                    let remainder_chunk = Rc::new(RefCell::new(MemorySuballocation {
                        start: alloc_end,
                        start_padding: 0,
                        key: remainder_key,
                        kind: MemoryKind::None,
                        prev: Some(chunk_value.clone()),
                        next: next.clone(),
                    }));
                    if let Some(next) = next {
                        next.borrow_mut().prev = Some(remainder_chunk.clone());
                    }
                    let none = self
                        .allocations
                        .insert(remainder_key, (remainder_chunk.clone(), memory.clone()));
                    assert!(none.is_none());

                    let key = MemorySuballocationKey {
                        availability_token: new_availability,
                        size: start_pad + size,
                        id: key.id,
                    };
                    chunk.next = Some(remainder_chunk);
                    chunk.key = key;
                    let none = self
                        .allocations
                        .insert(key, (chunk_value.clone(), memory.clone()));
                    assert!(none.is_none());
                }

                return Ok(SuballocationUgh {
                    memory,
                    offset: useful_start,
                    size,
                });
            }
        }

        let (allocation, info) = alloc_fun()?;
        assert!(info.size >= size);

        let id = self.make_new_id();
        let key = MemorySuballocationKey {
            availability_token: new_availability,
            size: info.size,
            id,
        };
        let chunk = Rc::new(RefCell::new(MemorySuballocation {
            start: 0,
            start_padding: 0,
            key,
            kind,
            prev: None,
            next: None,
        }));

        let memory = Rc::new(MemoryBlock {
            allocation,
            size: info.size,
            align,
            memory_type: memory_type_index,
            head: chunk.clone(),
        });

        self.allocations
            .insert(key, (chunk.clone(), memory.clone()));

        Ok(SuballocationUgh {
            memory,
            offset: 0,
            size,
        })
    }
    fn make_new_id(&mut self) -> u32 {
        let old = self.id_counter;
        self.id_counter += 1;
        old
    }
}

fn round_up_pow2(number: u64, multiple: u64) -> u64 {
    number.wrapping_add(multiple).wrapping_sub(1) & !multiple.wrapping_sub(1)
}

fn round_down_pow2(number: u64, multiple: u64) -> u64 {
    number & !multiple.wrapping_sub(1)
}
