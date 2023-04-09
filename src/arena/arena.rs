use std::{fmt::Debug, hash::Hash, marker::PhantomData, mem::ManuallyDrop, ops::Not};

use super::uint::{Config, Optional, PackedUint, UInt};

pub trait KeyType {
    const MAX_VALID: Self;
    const INVALID: Self;
}

pub trait KeyConfig: Config + Copy {
    const WRAP_GENERATION: bool;
}

pub trait Key: Copy {
    type Config: KeyConfig;
    type StoredIndex: UInt;
    type StoredGeneration: UInt;
    fn new(index: Self::StoredIndex, generation: Self::StoredGeneration) -> Self;
    fn index(&self) -> Self::StoredIndex;
    fn generation(&self) -> Self::StoredGeneration;
}

union EntryValue<K: Key, T> {
    next_free: Optional<K::StoredIndex>,
    value: ManuallyDrop<T>,
}

struct Entry<K: Key, T> {
    // even empty, odd occupied
    generation: K::StoredGeneration,
    value: EntryValue<K, T>,
}

impl<K: Key, T> Entry<K, T> {
    #[inline(always)]
    fn occupied(&self) -> bool {
        self.generation % UInt::from_usize(2) == UInt::from_usize(1)
    }
    #[inline(always)]
    fn take(self) -> Option<T> {
        self.occupied()
            .not()
            .then(|| unsafe { ManuallyDrop::into_inner(self.value.value) })
    }
}

pub struct GenArena<K: Key, T> {
    entries: Vec<Entry<K, T>>,
    next_free: Optional<K::StoredIndex>,
    #[allow(unused)]
    spooky: PhantomData<K>,
}

impl<K: Key, T> GenArena<K, T> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_free: Optional::NONE,
            spooky: PhantomData,
        }
    }
    pub fn get(&self, key: K) -> Option<&T> {
        unsafe { Some(&self.get_entry(key)?.value.value) }
    }
    pub fn get_mut(&mut self, key: K) -> Option<&mut T> {
        unsafe { Some(&mut self.get_entry_mut(key)?.value.value) }
    }
    fn get_entry(&self, key: K) -> Option<&Entry<K, T>> {
        let index = key.index();

        let entry = self.entries.get(index.into_usize())?;
        let generation = key.generation();

        if entry.generation != generation {
            return None;
        }

        Some(entry)
    }
    fn get_entry_mut(&mut self, key: K) -> Option<&mut Entry<K, T>> {
        let index = key.index();

        let entry = self.entries.get_mut(index.into_usize())?;
        let generation = key.generation();

        if entry.generation != generation {
            return None;
        }

        Some(entry)
    }
    pub fn insert(&mut self, value: T) -> K {
        if let Some(next) = self.next_free.get() {
            assert!(self.entries.len() - 1 < K::Config::MAX_FIRST);

            unsafe {
                let free = self.entries.get_unchecked_mut(next.into_usize());
                self.next_free = free.value.next_free;
                free.value.value = ManuallyDrop::new(value);
                free.generation = free.generation + 1.into();

                K::new(next, free.generation)
            }
        } else {
            let index = self.entries.len();
            self.entries.push(Entry {
                // odd generation for occupied
                generation: 1.into(),
                value: EntryValue {
                    value: ManuallyDrop::new(value),
                },
            });

            K::new(UInt::from_usize(index), 1.try_into().ok().unwrap())
        }
    }
    pub fn remove(&mut self, key: K) -> Option<T> {
        // can't use get_entry_mut here because borrowchk
        let index = key.index();

        let entry = self.entries.get(index.into_usize())?;
        let generation = key.generation();

        if entry.generation != generation {
            return None;
        };

        Some(unsafe { self.remove_internal(index) })
    }
    pub fn clear(&mut self) {
        for i in (0..self.entries.len()).rev() {
            let entry = unsafe { self.entries.get_unchecked_mut(i) };
            if entry.occupied() {
                unsafe { self.remove_internal(UInt::from_usize(i)) };
            }
        }
    }
    unsafe fn remove_internal(&mut self, index: <K as Key>::StoredIndex) -> T {
        let entry = self.entries.get_unchecked_mut(index.into_usize());
        entry.generation = entry.generation + 1.into();
        let value = std::mem::replace(
            &mut entry.value,
            EntryValue {
                next_free: self.next_free,
            },
        );
        let mut at_max = entry.generation >= UInt::from_usize(K::Config::MAX_SECOND);
        // we just stop using the slot if it reaches max generation
        if K::Config::WRAP_GENERATION && at_max {
            entry.generation = 0.into();
            at_max = false;
        }
        if !at_max {
            self.next_free = Optional::new_some(index);
        }
        ManuallyDrop::into_inner(unsafe { value.value })
    }
    pub fn contains(&self, key: K) -> bool {
        let index = key.index();

        let Some(entry) = self.entries.get(index.into_usize()) else {
            return false;
        };

        entry.generation == key.generation()
    }
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }
    pub fn total_capacity(&self) -> usize {
        self.entries.capacity()
    }
    pub fn iter(&self) -> Iter<K, T> {
        Iter {
            entries: self.entries.iter().enumerate(),
        }
    }
    pub fn iter_mut(&mut self) -> IterMut<K, T> {
        IterMut {
            entries: self.entries.iter_mut().enumerate(),
        }
    }
    pub fn into_iter(self) -> IntoIter<K, T> {
        IntoIter {
            entries: self.entries.into_iter(),
        }
    }
    pub fn drain_filter<F: FnMut(&mut T) -> bool>(&mut self, fun: F) -> DrainFilter<'_, K, T, F> {
        DrainFilter {
            arena: self,
            i: 0,
            fun,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U32KeyConfig;
impl Config for U32KeyConfig {
    const FIRST_BITS: usize = 15;
    const SECOND_BITS: usize = 17;
}
impl KeyConfig for U32KeyConfig {
    const WRAP_GENERATION: bool = true;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U32Key(PackedUint<U32KeyConfig, u32>);

impl Key for U32Key {
    type Config = U32KeyConfig;
    type StoredIndex = u32;
    type StoredGeneration = u32;

    fn new(index: Self::StoredIndex, generation: Self::StoredGeneration) -> Self {
        Self(PackedUint::new(index, generation))
    }
    fn index(&self) -> Self::StoredIndex {
        self.0.first()
    }
    fn generation(&self) -> Self::StoredGeneration {
        self.0.second()
    }
}

impl Debug for U32Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("U32Key")
            .field("index", &self.index().into_usize())
            .field("generation", &self.generation().into_usize())
            .finish()
    }
}

#[test]
fn test_arena() {
    // we want to test that entries are abandoned when they reach their maximum generation
    let max_by_one = UInt::from_usize(<U32Key as Key>::Config::MAX_SECOND);
    let mut map: GenArena<U32Key, u32> = GenArena {
        entries: vec![Entry {
            generation: max_by_one,
            value: EntryValue {
                value: ManuallyDrop::new(42),
            },
        }],
        next_free: Optional::NONE,
        spooky: PhantomData,
    };

    map.remove(Key::new(0, max_by_one));
    let inserted = map.insert(1);

    // test generation wraparound
    assert_eq!(map.entries[0].generation, 1);

    let removed = map.remove(inserted).unwrap();
    assert_eq!(removed, 1);

    map.insert(2);

    let none = map.remove(inserted);
    assert_eq!(none, None);
}

pub struct Iter<'a, K: Key, T> {
    entries: std::iter::Enumerate<std::slice::Iter<'a, Entry<K, T>>>,
}

impl<'a, K: Key, T> Iterator for Iter<'a, K, T> {
    type Item = (K, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((i, next)) = self.entries.next() {
            if next.occupied() {
                return Some((
                    K::new(K::StoredIndex::from_usize(i), next.generation),
                    // sound since they've been checked with occupied()
                    unsafe { &*next.value.value },
                ));
            }
        }
        return None;
    }
}

impl<'a, K: Key, T> DoubleEndedIterator for Iter<'a, K, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some((i, next)) = self.entries.next_back() {
            if next.occupied() {
                return Some((
                    K::new(K::StoredIndex::from_usize(i), next.generation),
                    // sound since they've been checked with occupied()
                    unsafe { &*next.value.value },
                ));
            }
        }
        return None;
    }
}

pub struct IterMut<'a, K: Key, T> {
    entries: std::iter::Enumerate<std::slice::IterMut<'a, Entry<K, T>>>,
}

impl<'a, K: Key, T> Iterator for IterMut<'a, K, T> {
    type Item = (K, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((i, next)) = self.entries.next() {
            if next.occupied() {
                return Some((
                    K::new(K::StoredIndex::from_usize(i), next.generation),
                    // sound since they've been checked with occupied()
                    unsafe { &mut *next.value.value },
                ));
            }
        }
        return None;
    }
}

impl<'a, K: Key, T> DoubleEndedIterator for IterMut<'a, K, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some((i, next)) = self.entries.next_back() {
            if next.occupied() {
                return Some((
                    K::new(K::StoredIndex::from_usize(i), next.generation),
                    // sound since they've been checked with occupied()
                    unsafe { &mut *next.value.value },
                ));
            }
        }
        return None;
    }
}

pub struct IntoIter<K: Key, T> {
    entries: std::vec::IntoIter<Entry<K, T>>,
}

impl<K: Key, T> Iterator for IntoIter<K, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(next) = self.entries.next() {
            let entry = next.take();
            if entry.is_some() {
                return entry;
            }
        }
        return None;
    }
}

impl<K: Key, T> DoubleEndedIterator for IntoIter<K, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some(next) = self.entries.next_back() {
            let entry = next.take();
            if entry.is_some() {
                return entry;
            }
        }
        return None;
    }
}

impl<'a, K: Key, T> IntoIterator for &'a GenArena<K, T> {
    type Item = (K, &'a T);
    type IntoIter = Iter<'a, K, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Key, T> IntoIterator for &'a mut GenArena<K, T> {
    type Item = (K, &'a mut T);
    type IntoIter = IterMut<'a, K, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K: Key, T> IntoIterator for GenArena<K, T> {
    type Item = T;
    type IntoIter = IntoIter<K, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_iter()
    }
}

pub struct DrainFilter<'a, K: Key, T, F: FnMut(&mut T) -> bool> {
    arena: &'a mut GenArena<K, T>,
    i: usize,
    fun: F,
}

impl<'a, K: Key, T, F: FnMut(&mut T) -> bool> Iterator for DrainFilter<'a, K, T, F> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.arena.entries.get_mut(self.i) {
            if entry.occupied() {
                // both sound since they've been checked with occupied()
                let val = unsafe { &mut *entry.value.value };
                if (self.fun)(val) {
                    return Some(unsafe { self.arena.remove_internal(UInt::from_usize(self.i)) });
                }
            }
            self.i += 1;
        }
        return None;
    }
}
