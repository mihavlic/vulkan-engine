use std::ops::{Add, BitAnd, BitOr, Div, Mul, Rem, Shl, Shr};

pub trait UInt:
    Sized
    + Copy
    + PartialOrd
    + Ord
    + Eq
    + Add<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + From<u8>
{
    const MAX: Self;
    fn into_usize(&self) -> usize;
    fn from_usize(value: usize) -> Self;
}

macro_rules! impl_int {
    ($($name:ident),+) => {
        $(
            impl UInt for $name {
                const MAX: Self = $name::MAX;
                #[inline(always)]
                fn into_usize(&self) -> usize {
                    <Self as TryInto<usize>>::try_into(*self).unwrap()
                }
                #[inline(always)]
                fn from_usize(value: usize) -> Self {
                    <Self as TryFrom<usize>>::try_from(value).unwrap()
                }
            }
        )+
    }
}

impl_int!(u8, u16, u32, u64, u128, usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Optional<T: UInt>(T);

impl<T: UInt> Optional<T> {
    pub const NONE: Self = Self(T::MAX);

    pub fn new(value: T) -> Optional<T> {
        assert!(value != T::MAX);
        Self(value)
    }
    pub fn new_none() -> Self {
        Self::NONE
    }
    pub fn get(&self) -> Option<T> {
        if self.0 == T::MAX {
            None
        } else {
            Some(self.0)
        }
    }
    pub fn is_some(&self) -> bool {
        self.0 != T::MAX
    }
    pub fn is_none(&self) -> bool {
        self.0 == T::MAX
    }
    fn set(&mut self, value: Option<T>) {
        if let Some(value) = value {
            assert!(value != T::MAX);
            *self = Self(value);
        } else {
            *self = Self::NONE;
        }
    }
}

pub type OptionalU32 = Optional<u32>;
