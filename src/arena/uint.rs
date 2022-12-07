use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Add, BitAnd, BitOr, Div, Mul, Rem, Shl, Shr},
};

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
    + Display
    + Debug
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

    #[inline]
    pub fn new(value: Option<T>) -> Optional<T> {
        assert!(value != Some(T::MAX));
        let value = value.unwrap_or(T::MAX);
        Self(value)
    }
    #[inline]
    pub fn new_some(value: T) -> Optional<T> {
        assert!(value != T::MAX);
        Self(value)
    }
    #[inline]
    pub fn new_none() -> Self {
        Self::NONE
    }
    #[inline]
    pub fn get(&self) -> Option<T> {
        if self.0 == T::MAX {
            None
        } else {
            Some(self.0)
        }
    }
    #[inline]
    pub fn is_some(&self) -> bool {
        self.0 != T::MAX
    }
    #[inline]
    pub fn is_none(&self) -> bool {
        self.0 == T::MAX
    }
    #[inline]
    pub fn set(&mut self, value: Option<T>) {
        if let Some(value) = value {
            assert!(value != T::MAX);
            *self = Self(value);
        } else {
            *self = Self::NONE;
        }
    }
}

pub type OptionalU32 = Optional<u32>;

pub trait Config: Copy {
    const FIRST_BITS: usize;
    const SECOND_BITS: usize;

    const MAX_FIRST: usize = 2usize.pow(Self::FIRST_BITS as u32) - 1;
    const MAX_SECOND: usize = 2usize.pow(Self::SECOND_BITS as u32) - 1;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackedUint<C: Config, I: UInt>(I, PhantomData<C>);

impl<C: Config, I: UInt> PackedUint<C, I> {
    pub const BITS: usize = std::mem::size_of::<I>() * 8;
    const __ASSERT: () = assert!(C::FIRST_BITS + C::SECOND_BITS <= Self::BITS);

    #[rustfmt::skip]
    pub(crate) fn new(first: I, second: I) -> Self {
        debug_assert!(first <= UInt::from_usize(C::MAX_FIRST), "First is too large {} > {}", first, C::MAX_FIRST);
        debug_assert!(second <= UInt::from_usize(C::MAX_SECOND), "Second is too large {} > {}", second, C::MAX_SECOND);

        // little:
        // [first][second]

        // big:
        // [second][first]

        #[cfg(target_endian = "little")]
        let value =
              (first << Self::BITS - C::FIRST_BITS)
            | (second & (I::MAX >> Self::BITS - C::SECOND_BITS));

        #[cfg(target_endian = "big")]
        let value =
              (first >> Self::BITS - C::FIRST_BITS)
            | (second & (I::MAX << Self::BITS - C::SECOND_BITS));

        PackedUint(value, PhantomData)
    }

    pub(crate) fn first(&self) -> I {
        #[cfg(target_endian = "little")]
        let value = self.0 >> (Self::BITS - C::FIRST_BITS);

        #[cfg(target_endian = "big")]
        let value = self.0 << (Self::BITS - C::FIRST_BITS);

        value
    }

    pub(crate) fn second(&self) -> I {
        #[cfg(target_endian = "little")]
        let value = self.0 & (I::MAX >> Self::BITS - C::SECOND_BITS);

        #[cfg(target_endian = "big")]
        let value = self.0 & (I::MAX << Self::BITS - C::SECOND_BITS);

        value
    }
}

impl<C: Config, I: UInt> Debug for PackedUint<C, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedUint")
            .field("first", &self.first())
            .field("second", &self.second())
            .finish()
    }
}
