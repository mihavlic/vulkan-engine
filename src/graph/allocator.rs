use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

use crate::simple_handle;

use super::GraphResource;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    Linear,
    NonLinear,
    None,
}

#[derive(Clone, Copy)]
pub struct MemoryChunk {
    // Some means allocation, None free space
    allocation: Option<BlockAllocation>,
    // start is always the end of the previous block
    end: u64,
    kind: MemoryKind,
}

pub struct MemoryBlock {
    size: u64,
    align: u64,
    chunks: Vec<MemoryChunk>,
    total_free: u64,
    allocations_counter: u32,
    allocation: pumice_vma::Allocation,
}

impl MemoryBlock {
    pub fn new(size: u64, align: u64, allocation: pumice_vma::Allocation) -> Self {
        Self {
            size,
            align,
            chunks: vec![MemoryChunk {
                allocation: None,
                end: size,
                kind: MemoryKind::None,
            }],
            total_free: size,
            allocations_counter: u32::MAX,
            allocation,
        }
    }
    // pub fn new_used(size: u64, align: u64, kind: MemoryKind) -> (Self, BlockAllocation) {
    //     (
    //         Self {
    //             size,
    //             align,
    //             chunks: vec![MemoryChunk {
    //                 allocation: Some(BlockAllocation(0)),
    //                 end: size,
    //                 kind,
    //             }],
    //             total_free: 0,
    //             allocations_counter: 1,
    //         },
    //         BlockAllocation(0),
    //     )
    // }
    pub fn free(&mut self, allocation: BlockAllocation) {
        let chunks = &self.chunks;

        let pos = chunks
            .iter()
            .position(|c| c.allocation == Some(allocation))
            .expect("Allocation not found");

        let (start, end) = self.get_chunk_start_end(pos);
        self.total_free += end - start;

        let prev = pos
            .checked_sub(1)
            .and_then(|c| chunks.get(c))
            .and_then(|c| c.allocation.is_none().then_some(()))
            .is_some();
        let next = pos
            .checked_add(1)
            .and_then(|c| chunks.get(c))
            .and_then(|c| c.allocation.is_none().then_some(()))
            .is_some();

        let (start, end) = match (prev, next) {
            (true, true) => (pos - 1, pos + 1),
            (true, false) => (pos - 1, pos),
            (false, true) => (pos, pos + 1),
            (false, false) => (pos, pos),
        };

        let new = MemoryChunk {
            allocation: None,
            end: chunks[end].end,
            kind: MemoryKind::None,
        };

        self.chunks[start] = new;
        // for the (pos, pos) case this will create a (1..=0) range, which is considered empty since `start > end`
        self.chunks.drain(start + 1..=end);
    }
    fn get_chunk_start_end(&self, chunk: usize) -> (u64, u64) {
        let start = if chunk == 0 {
            0
        } else {
            self.chunks[chunk - 1].end
        };
        (start, self.chunks[chunk].end)
    }
    // returns a handle which may be used to later allocate the best matching block or memory, with the amount of memory remaining on the right side after the would-be allocation
    pub fn consider_allocate(
        &self,
        size: u64,
        align: u64,
        kind: MemoryKind,
        linear_nonlinear_granularity: u64,
        best_fit: bool,
    ) -> Option<(ConsiderAllocation, u64)> {
        assert!(size > 0);
        assert!(align > 0);

        if size > self.total_free {
            return None;
        }

        let mut chunks = self.chunks.iter().enumerate();
        let mut chunks = std::iter::from_fn(|| {
            while let Some((i, c)) = chunks.next() {
                // the chunk is already taken
                if c.allocation.is_some() {
                    continue;
                }

                let (c_start, c_end) = self.get_chunk_start_end(i);

                // fast reject
                if c_end - c_start < size {
                    continue;
                }

                let (start, end) = self.suballocation_compute_start_end(
                    i,
                    align,
                    kind,
                    linear_nonlinear_granularity,
                );

                // round has put us past the end of the allocation
                if start >= end {
                    continue;
                }

                // allocation can't fit in this chunk
                if end - start < size {
                    continue;
                }

                return Some((ConsiderAllocation::new(i), (c_end - c_start) - size));
            }
            None
        });

        if !best_fit {
            return chunks.next();
        }

        let mut adj_block_rem = u64::MAX;
        let mut chunk = None;
        for (i, rem) in chunks {
            if adj_block_rem > rem {
                adj_block_rem = rem;
                chunk = Some(i);
            }
        }

        chunk.map(|c| (c, adj_block_rem))
    }
    fn suballocation_compute_start_end(
        &self,
        i: usize,
        align: u64,
        kind: MemoryKind,
        linear_nonlinear_granularity: u64,
    ) -> (u64, u64) {
        let mut alignment = align;
        if let Some(prev) = i.checked_sub(1).and_then(|c| self.chunks.get(c)) {
            assert!(prev.kind != MemoryKind::None);
            if kind != prev.kind {
                alignment = alignment.max(linear_nonlinear_granularity);
            }
        }

        let (start, mut end) = self.get_chunk_start_end(i);
        // check if we need to care about `bufferImageGranularity` if so, round the end down to keep clear of the next resource's page
        if let Some(next) = self.chunks.get(i + 1) {
            assert!(next.kind != MemoryKind::None);
            if kind != next.kind {
                end = round_down_pow2(end, linear_nonlinear_granularity);
            }
        }

        // round up to reach alignment
        let start = round_up_pow2(start, align as u64);
        (start, end)
    }
    fn make_allocation_handle(&mut self) -> BlockAllocation {
        self.allocations_counter = self.allocations_counter.wrapping_add(1);
        BlockAllocation(self.allocations_counter)
    }
    pub fn allocate_considered(
        &mut self,
        considered: ConsiderAllocation,
        size: u64,
        align: u64,
        kind: MemoryKind,
        linear_nonlinear_granularity: u64,
    ) -> BlockAllocation {
        let i = considered.index();
        assert!(self.chunks[i].allocation.is_none());
        assert!(self.chunks[i].kind == MemoryKind::None);

        // recompute the start and end offsets
        let (start, end) =
            self.suballocation_compute_start_end(i, align, kind, linear_nonlinear_granularity);
        // do some sanity checking
        assert!(start < end);
        assert!(end - start >= size);

        self.total_free -= size;

        let alloc_end = start + size;
        let allocation = self.make_allocation_handle();
        let alloc = MemoryChunk {
            allocation: Some(allocation),
            end: alloc_end,
            kind,
        };

        let (original_start, original_end) = self.get_chunk_start_end(i);

        if alloc_end == original_end {
            self.chunks[i] = alloc;
        } else {
            self.chunks.insert(i, alloc);
        }

        if start != original_start {
            self.chunks.insert(
                i,
                MemoryChunk {
                    allocation: None,
                    end: start,
                    kind: MemoryKind::None,
                },
            );
        }

        allocation
    }
    pub fn allocate(
        &mut self,
        size: u64,
        align: u64,
        kind: MemoryKind,
        linear_nonlinear_granularity: u64,
    ) -> Option<BlockAllocation> {
        let (considered, _) =
            self.consider_allocate(size, align, kind, linear_nonlinear_granularity, true)?;
        Some(self.allocate_considered(considered, size, align, kind, linear_nonlinear_granularity))
    }
    // returns the largest size of free blocks, this is without any alignment so may still be insuficient for an allocation of the same size
    pub fn compute_max_free_size(&self) -> Option<u64> {
        let mut max = 0;
        for (i, c) in self.chunks.iter().enumerate() {
            if c.allocation.is_some() {
                continue;
            }
            let (start, end) = self.get_chunk_start_end(i);
            let size = end - start;
            max = max.max(size);
        }
        (max != 0).then_some(max)
    }
    pub fn compute_min_free_size(&self) -> Option<u64> {
        let mut max = u64::MAX;
        for (i, c) in self.chunks.iter().enumerate() {
            if c.allocation.is_some() {
                continue;
            }
            let (start, end) = self.get_chunk_start_end(i);
            let size = end - start;
            max = max.max(size);
        }
        (max != u64::MAX).then_some(max)
    }
    pub fn get_allocation_size(&self, allocation: BlockAllocation) -> u64 {
        let (start, end) = self.get_chunk_start_end(allocation.index());
        end - start
    }
}

pub fn round_up_pow2(number: u64, multiple: u64) -> u64 {
    number.wrapping_add(multiple).wrapping_sub(1) & !multiple.wrapping_sub(1)
}

pub fn round_down_pow2(number: u64, multiple: u64) -> u64 {
    number & !multiple.wrapping_sub(1)
}

simple_handle! {
    pub ConsiderAllocation,
    pub BlockAllocation
}

#[test]
pub fn test_block_allocator() {
    let mut block = MemoryBlock::new(64, 64, pumice_vma::Allocation::null());
    let (_, rem) = block
        .consider_allocate(64, 64, MemoryKind::Linear, 1, true)
        .unwrap();
    assert_eq!(rem, 0);
    let alloc = block.allocate(64, 64, MemoryKind::Linear, 1);
    assert!(alloc.is_some());
}

// structure managing multiple blocks, allowing to efficiently make allocations into them

pub struct AllocatorAllocation {
    allocation: BlockAllocation,
    node: Rc<BlockAllocatorNode>,
    resource: GraphResource,
}

pub struct BlockAllocatorNode {
    min_size: Cell<u64>,
    // always increasing, only used to give blocks a unique identity that isn't dependant on their min_size
    id: u32,
    // generally always Some(), Option is only used to construct a dummy node to iterate over larger actual nodes
    block: RefCell<Option<MemoryBlock>>,
}

impl PartialOrd for BlockAllocatorNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.min_size.partial_cmp(&other.min_size) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.id.partial_cmp(&other.id)
        // ignore self.block
    }
}

impl Ord for BlockAllocatorNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.min_size
            .cmp(&other.min_size)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialEq for BlockAllocatorNode {
    fn eq(&self, other: &Self) -> bool {
        self.min_size == other.min_size && self.id == other.id
    }
}

impl Eq for BlockAllocatorNode {}

pub struct BlockAllocator {
    // crimes
    // we're putting all the state in the key because reasons
    // don't be afraid we'll be careful not to modify the ordering while inserted within the map
    // we need a BTreeMap rather than *Set because we need the remove_entry function
    blocks: BTreeMap<Rc<BlockAllocatorNode>, ()>,
    monotonic: u32,
}

impl BlockAllocator {
    pub fn new() -> Self {
        Self {
            blocks: BTreeMap::new(),
            monotonic: 0,
        }
    }
    pub fn free(&mut self, allocation: AllocatorAllocation) {
        let AllocatorAllocation {
            allocation,
            node,
            resource: _,
        } = allocation;
        let block = RefMut::map(node.block.borrow_mut(), |opt| opt.as_mut().unwrap());
        let size = block.get_allocation_size(allocation);
        block.free(allocation);

        if size > node.min_size.get() {
            // we have not decreased the min_size of the block
            // thus the ordering is unchanged and we don't need to reinstert the node
        } else {
            self.update_node_ordering(&node, &*block);
        }
    }
    pub fn allocate<
        F1: FnOnce(u64, u64, MemoryKind, u64) -> MemoryBlock,
        F2: FnMut(&AllocatorAllocation) -> bool,
    >(
        &mut self,
        size: u64,
        align: u64,
        kind: MemoryKind,
        linear_nonlinear_granularity: u64,
        new_block_fun: F1,
        block_filter: F2,
        resource: GraphResource,
    ) -> AllocatorAllocation {
        let dummy = BlockAllocatorNode {
            min_size: Cell::new(size),
            id: 0,
            block: RefCell::new(None),
        };

        // even though the blocks have a sufficient min_size, the allocation may still not fit due to alignment padding and such
        for (node, _) in self.blocks.range(dummy..).filter(|&(b, _)| {
            block_filter(&*Ref::map(b.block.borrow(), |opt| opt.as_ref().unwrap()))
        }) {
            let block = RefMut::map(node.block.borrow_mut(), |opt| opt.as_mut().unwrap());
            // TODO maybe do something fancier than first-fit
            if let Some(alloc) = block.allocate(size, align, kind, linear_nonlinear_granularity) {
                if size > node.min_size.get() {
                    // we have not decreased the min_size of the block
                    // thus the ordering is unchanged and we don't need to reinstert the node
                } else {
                    self.update_node_ordering(node, &*block);
                }

                return AllocatorAllocation {
                    allocation: alloc,
                    node: node.clone(),
                    resource,
                };
            }
        }

        // we've failed to find any space in the existing blocks, we need to make another one
        let block = new_block_fun(size, align, kind, linear_nonlinear_granularity);
        let alloc = block
            .allocate(size, align, kind, linear_nonlinear_granularity)
            .expect("new_block_fun must create a sufficient memory block");

        let size = block.compute_min_free_size().unwrap_or(0);
        let node = Rc::new(BlockAllocatorNode {
            min_size: Cell::new(size),
            id: self.monotonic,
            block: RefCell::new(Some(block)),
        });

        let none = self.blocks.insert(node.clone(), ());
        assert!(none.is_none(), "Nodes must be unique");
        let handle = AllocatorAllocation {
            allocation: alloc,
            node: node,
            resource,
        };

        self.monotonic = self
            .monotonic
            .checked_add(1)
            .expect("Allocation counter has overflowed, TODO handle this?");

        handle
    }
    fn update_node_ordering(&mut self, node: &Rc<BlockAllocatorNode>, block: &MemoryBlock) {
        // we need to remove the entry, modify its ordering and then reinsert it, perfectly safe!
        let (data, _) = self
            .blocks
            .remove_entry(node)
            .expect("The memory block is not withing the allocator!");
        let size = block.compute_min_free_size().unwrap_or(0);
        data.min_size.set(size);
        self.blocks.insert(data, ());
    }
}

impl Default for BlockAllocator {
    fn default() -> Self {
        Self::new()
    }
}
