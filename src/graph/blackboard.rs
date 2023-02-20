use std::{
    any::{Any, TypeId},
    collections::hash_map::Entry,
};

use crate::storage::constant_ahash_hashmap;

pub struct BlackBoard {
    map: ahash::HashMap<TypeId, Box<dyn Any>>,
}

impl BlackBoard {
    pub fn new() -> Self {
        Self {
            map: constant_ahash_hashmap(),
        }
    }
    pub fn add<T: 'static>(&mut self, value: T) {
        self.map.insert(TypeId::of::<T>(), Box::new(value));
    }
    pub fn get<T: 'static>(&self) -> Option<&T> {
        let any = self.map.get(&TypeId::of::<T>())?;
        let downcast = any.downcast_ref::<T>().unwrap();
        Some(downcast)
    }
    pub fn get_or_insert_with<T: 'static, F: FnOnce() -> T>(&mut self, fun: F) -> &mut T {
        let any = self
            .map
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(fun()));
        any.downcast_mut::<T>().unwrap()
    }
    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        let any = self.map.get_mut(&TypeId::of::<T>())?;
        let downcast = any.downcast_mut::<T>().unwrap();
        Some(downcast)
    }
}
