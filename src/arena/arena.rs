use std::{fmt::Debug, hash::Hash, marker::PhantomData, mem::ManuallyDrop, ops::Not};

use super::optional::{Optional, UInt};

pub trait KeyType {
    const MAX_VALID: Self;
    const INVALID: Self;
}

pub trait KeyConfig: Copy {
    const INDEX_BITS: usize;
    const GENERATION_BITS: usize;
    const WRAP_GENERATION: bool;

    const MAX_INDEX: usize = 2usize.pow(Self::INDEX_BITS as u32);
    const MAX_GENERATION: usize = 2usize.pow(Self::GENERATION_BITS as u32);
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
            assert!(self.entries.len() - 1 < K::Config::MAX_INDEX);

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
                generation: 0.into(),
                value: EntryValue {
                    value: ManuallyDrop::new(value),
                },
            });

            K::new(UInt::from_usize(index), 0.try_into().ok().unwrap())
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
    unsafe fn remove_internal(&mut self, index: <K as Key>::StoredIndex) -> T {
        let entry = self.entries.get_unchecked_mut(index.into_usize());
        entry.generation = entry.generation + 1.into();
        let value = std::mem::replace(
            &mut entry.value,
            EntryValue {
                next_free: self.next_free,
            },
        );
        let mut at_max = entry.generation >= UInt::from_usize(K::Config::MAX_GENERATION);
        // we just stop using the slot if it reaches max generation
        if K::Config::WRAP_GENERATION && at_max {
            entry.generation = 0.into();
            at_max = false;
        }
        if !at_max {
            self.next_free = Optional::new(index);
        }
        ManuallyDrop::into_inner(unsafe { value.value })
    }
    pub fn contains(&self, key: K) -> bool {
        let index = key.index();

        let Some(entry) = self.entries.get(index.into_usize()) else {
            return false;
        };

        let generation = key.generation();
        entry.generation == generation
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
pub struct IntKey<C: KeyConfig, I: UInt>(I, PhantomData<C>);

impl<C: KeyConfig, I: UInt> IntKey<C, I> {
    const BITS: usize = std::mem::size_of::<I>() * 8;
    const __ASSERT: () = assert!(C::INDEX_BITS + C::GENERATION_BITS <= Self::BITS);
}

impl<C: KeyConfig, I: UInt> Key for IntKey<C, I> {
    type Config = C;
    type StoredIndex = I;
    type StoredGeneration = I;

    fn new(index: I, generation: I) -> Self {
        debug_assert!(index <= UInt::from_usize(C::MAX_INDEX));
        debug_assert!(generation <= UInt::from_usize(C::MAX_GENERATION));

        // little:
        // [generation][index]

        // big:
        // [index][generation]

        #[cfg(target_endian = "little")]
        let value = (index & (I::MAX >> Self::BITS - C::INDEX_BITS))
            | (generation << Self::BITS - C::GENERATION_BITS);

        #[cfg(target_endian = "big")]
        let value = (index & (I::MAX << Self::BITS - C::INDEX_BITS))
            | (generation >> Self::BITS - C::GENERATION_BITS);

        IntKey(value, PhantomData)
    }

    fn index(&self) -> I {
        #[cfg(target_endian = "little")]
        let value = self.0 & (I::MAX >> C::GENERATION_BITS);

        #[cfg(target_endian = "big")]
        let value = self.0 & (I::MAX << C::GENERATION_BITS);

        value
    }

    fn generation(&self) -> I {
        #[cfg(target_endian = "little")]
        let value = self.0 >> (Self::BITS - C::GENERATION_BITS);

        #[cfg(target_endian = "big")]
        let value = self.0 << (Self::BITS - C::GENERATION_BITS);

        value
    }
}

impl<C: KeyConfig, I: UInt> Debug for IntKey<C, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntKey")
            .field("index", &self.index().into_usize())
            .field("generation", &self.generation().into_usize())
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U32KeyConfig;
impl KeyConfig for U32KeyConfig {
    const INDEX_BITS: usize = 15;
    const GENERATION_BITS: usize = 17;
    const WRAP_GENERATION: bool = true;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U32Key(IntKey<U32KeyConfig, u32>);

impl Key for U32Key {
    type Config = U32KeyConfig;
    type StoredIndex = u32;
    type StoredGeneration = u32;

    fn new(index: Self::StoredIndex, generation: Self::StoredGeneration) -> Self {
        Self(IntKey::new(index, generation))
    }
    fn index(&self) -> Self::StoredIndex {
        IntKey::index(&self.0)
    }
    fn generation(&self) -> Self::StoredGeneration {
        IntKey::generation(&self.0)
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
    let max_by_one = UInt::from_usize(<U32Key as Key>::Config::MAX_GENERATION - 1);
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

    map.remove(<U32Key as Key>::new(0, max_by_one));
    let inserted = map.insert(1);

    assert_eq!(map.capacity(), 2);

    let removed = map.remove(inserted).unwrap();
    assert_eq!(removed, 1);

    map.insert(2);
    assert_eq!(map.capacity(), 2);

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
