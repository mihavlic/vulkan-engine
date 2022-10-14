use std::{
    cell::Cell,
    fmt::{Display, Write},
};

pub struct IterDisplay<
    T: IntoIterator,
    F: Fn(T::Item, &mut std::fmt::Formatter<'_>) -> std::fmt::Result,
> {
    iter: Cell<Option<T>>,
    fun: F,
}

impl<T: IntoIterator, F: Fn(T::Item, &mut std::fmt::Formatter<'_>) -> std::fmt::Result>
    IterDisplay<T, F>
{
    pub fn new(iter: T, fun: F) -> Self {
        Self {
            iter: Cell::new(Some(iter)),
            fun,
        }
    }
}

impl<T: IntoIterator, F: Fn(T::Item, &mut std::fmt::Formatter<'_>) -> std::fmt::Result> Display
    for IterDisplay<T, F>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for i in self.iter.replace(None).unwrap() {
            if !first {
                f.write_char('\n')?;
            }
            (self.fun)(i, f)?;
            first = false;
        }
        Ok(())
    }
}

pub struct Fun<F: FnOnce(&mut std::fmt::Formatter<'_>) -> std::fmt::Result>(Cell<Option<F>>);

impl<F: FnOnce(&mut std::fmt::Formatter<'_>) -> std::fmt::Result> Fun<F> {
    pub fn new(fun: F) -> Self {
        Self(Cell::new(Some(fun)))
    }
}

impl<F: FnOnce(&mut std::fmt::Formatter<'_>) -> std::fmt::Result> Display for Fun<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (self.0.replace(None).unwrap())(f)
    }
}
