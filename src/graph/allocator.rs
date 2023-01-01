use crate::simple_handle;

struct MemoryBlock {
    allocation: pumice_vma::Allocation,
    // info: pumice_vma::,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MemoryKind {
    Linear,
    NonLinear,
    None,
}

#[derive(Clone, Copy)]
struct MemoryChunk {
    // Some means allocation, None free space
    allocation: Option<BlockAllocation>,
    // start is always the end of the previous block
    end: u64,
    kind: MemoryKind,
}

struct Block {
    /* memory_heap: u32,
    memory_type: u32, */
    size: u64,
    align: u64,
    chunks: Vec<MemoryChunk>,
    total_free: u64,
    allocations_counter: u32,
}

impl Block {
    fn new(size: u64, align: u64 /* memory_heap: u32, memory_type: u32 */) -> Self {
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
        }
    }
    fn new_used(size: u64, align: u64, kind: MemoryKind) -> (Self, BlockAllocation) {
        (
            Self {
                size,
                align,
                chunks: vec![MemoryChunk {
                    allocation: Some(BlockAllocation(0)),
                    end: size,
                    kind,
                }],
                total_free: 0,
                allocations_counter: 1,
            },
            BlockAllocation(0),
        )
    }
    fn free(&mut self, allocation: BlockAllocation) {
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
    fn consider_allocate(
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
    fn allocate_considered(
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
    fn allocate(
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
}

fn round_up_pow2(number: u64, multiple: u64) -> u64 {
    number.wrapping_add(multiple).wrapping_sub(1) & !multiple.wrapping_sub(1)
}

fn round_down_pow2(number: u64, multiple: u64) -> u64 {
    number & !multiple.wrapping_sub(1)
}

simple_handle! {
    pub ConsiderAllocation,
    pub BlockAllocation
}

#[test]
fn test_block_allocator() {
    let mut block = Block::new(64, 64);
    let (_, rem) = block
        .consider_allocate(64, 64, MemoryKind::Linear, 1, true)
        .unwrap();
    assert_eq!(rem, 0);
    let alloc = block.allocate(64, 64, MemoryKind::Linear, 1);
    assert!(alloc.is_some());
}
