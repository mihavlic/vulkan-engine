// pub trait KeyExtractor {
//     type Source;
//     type Key: Ord;
//     pub fn extract_key(src: &Self::Source) -> Self::Key;
// }

// macro_rules! impl_extract {
//     ($($name:ident ($first:ident, $($other:ident),+))+) => {
//         $(
//             pub struct $name/* <$first, $($other),+>(PhantomData<$first>, $(PhantomData<$other>),+) */;
//             impl<$first: Ord + Copy, $($other),+> KeyExtractor for $name/* <$first, $($other),+> */ {
//                 type Source = ($first, $($other),+);
//                 type Key = $first;
//                 pub fn extract_key(src: &Self::Source) -> Self::Key {
//                     src.0
//                 }
//             }
//         )+
//     };
// }

// impl_extract! {
//     Tuple2 (A, B)
//     Tuple3 (A, B, C)
//     Tuple4 (A, B, C, D)
//     Tuple5 (A, B, C, D, E)
// }

pub trait KeyExtract {
    type Key: Ord;
    fn extract_key(&self) -> Self::Key;
}

macro_rules! impl_extract {
    ($(($first:ident, $($other:ident),+))+) => {
        $(
            impl<$first: Ord + Copy, $($other),+> KeyExtract for ($first, $($other),+) {
                type Key = $first;
                fn extract_key(&self) -> Self::Key {
                    self.0
                }
            }
        )+
    };
}

impl_extract! {
    (A, B)
    (A, B, C)
    (A, B, C, D)
    (A, B, C, D, E)
}

#[derive(Clone)]
pub struct LazySorted<T>(Vec<T>, bool);

impl<T> Default for LazySorted<T> {
    fn default() -> Self {
        Self(Default::default(), false)
    }
}

impl<T: KeyExtract> LazySorted<T> {
    pub fn push(&mut self, val: T) {
        self.1 = true;
        self.0.push(val)
    }
    pub fn sort_unstable_by_key(&mut self) {}
    pub fn sort_unstable(&mut self) {}
    pub fn ensure_sorted(&mut self) {
        if self.1 {
            self.0.sort_unstable_by_key(T::extract_key);
            self.1 = false;
        }
    }
    pub fn first(&mut self) -> Option<&T> {
        self.ensure_sorted();
        self.0.first()
    }
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.ensure_sorted();
        self.0.first_mut()
    }
    pub fn last(&mut self) -> Option<&T> {
        self.ensure_sorted();
        self.0.last()
    }
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.ensure_sorted();
        self.0.last_mut()
    }
    pub fn as_slice(&mut self) -> &[T] {
        self.ensure_sorted();
        self.0.as_slice()
    }
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.ensure_sorted();
        self.0.as_mut_slice()
    }
    pub fn get(&mut self, index: usize) -> Option<&T> {
        self.ensure_sorted();
        self.0.get(index)
    }
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.ensure_sorted();
        self.0.get_mut(index)
    }
    pub fn iter(&mut self) -> std::slice::Iter<'_, T> {
        self.ensure_sorted();
        self.0.iter()
    }
}

impl<T: KeyExtract> IntoIterator for LazySorted<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(mut self) -> Self::IntoIter {
        self.ensure_sorted();
        self.0.into_iter()
    }
}

impl<A> FromIterator<A> for LazySorted<A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        Self(Vec::from_iter(iter), true)
    }
}
