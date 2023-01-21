use std::fmt::{Display, Write};

use smallvec::{smallvec, SmallVec};

#[macro_export]
macro_rules! token_abuse {
    ($writer:expr, ($($var:expr),*) $($toks:tt)+) => {
        use std::borrow::BorrowMut;
       $crate::util::macro_abuse::write_with_interpolations($writer.borrow_mut(), stringify!($($toks)+), true, &[$(&$var),*]);
    };
    ($writer:expr, $str:expr, $($var:expr),*) => {
        use std::borrow::BorrowMut;
        $crate::util::macro_abuse::write_with_interpolations($writer.borrow_mut(), $str, false, &[$(&$var),*]);
    };
}

#[doc(hidden)]
pub fn write_with_interpolations(
    w: &mut dyn std::fmt::Write,
    str: &str,
    supress_lf: bool,
    interp: &[&dyn Display],
) {
    let mut interp = interp.into_iter();
    let mut chars = str.chars();
    while let Some(mut c) = chars.next() {
        match c {
            '$' => {
                let i = interp.next().unwrap();
                write!(w, "{i}").unwrap();
                continue;
            }
            '\n' | '\r' if supress_lf => continue,
            '\\' if chars.clone().next() == Some('$') => {
                chars.next();
                c = '$';
            }
            _ => {}
        }

        write!(w, "{c}").unwrap();
    }
}

#[repr(u8)]
enum State {
    Normal,
    SupressNewlines,
}

#[derive(PartialEq, Eq)]
enum Transient {
    Normal,
    Newline,
}

pub struct WeirdFormatter<'a> {
    w: &'a mut dyn std::io::Write,
    stack: SmallVec<[State; 16]>,
    transient: Transient,
    sep: bool,
    ident: u32,
}

impl<'a> WeirdFormatter<'a> {
    pub fn new(writer: &'a mut dyn std::io::Write) -> Self {
        Self {
            w: writer,
            stack: smallvec![State::Normal],
            transient: Transient::Normal,
            sep: false,
            ident: 0,
        }
    }
    pub fn write_char_io(&mut self, c: char) -> std::io::Result<()> {
        let Self {
            w,
            stack,
            transient,
            sep,
            ident,
        } = self;

        if *transient == Transient::Newline && matches!(stack.last(), Some(State::SupressNewlines))
        {
            *transient = Transient::Normal;
        }

        let mut t = Transient::Normal;
        let mut delayed_ident = 0;
        match c {
            '{' => {
                delayed_ident = 1;
                *sep = true;
                t = Transient::Newline;
                stack.push(State::Normal);
            }
            '}' => {
                *ident = ident.saturating_sub(1);
                *transient = Transient::Newline;
                t = Transient::Newline;
                stack.pop();
            }
            '(' => {
                stack.push(State::SupressNewlines);
            }
            ')' => {
                *sep = false;
                stack.pop();
            }
            '[' => {
                stack.push(State::SupressNewlines);
            }
            ']' => {
                *sep = false;
                stack.pop();
            }
            ';' => {
                *sep = false;
                t = Transient::Newline;
            }
            ' ' => {
                *sep = true;
                return Ok(());
            }
            '\n' => {
                if *transient == Transient::Newline {
                    write!(w, "\n");
                }

                *transient = Transient::Newline;
                return Ok(());
            }
            _ => {}
        };

        if *transient == Transient::Newline && matches!(stack.last(), Some(State::Normal)) {
            *sep = false;
            write!(w, "\n")?;
            for _ in 0..*ident {
                write!(w, "    ")?;
            }
        }

        *transient = t;
        *ident += delayed_ident;

        if *sep {
            *sep = false;
            write!(w, " ")?;
        }

        write!(w, "{c}")
    }
}

impl<'a> std::fmt::Write for WeirdFormatter<'a> {
    fn write_char(&mut self, c: char) -> std::fmt::Result {
        self.write_char_io(c).map_err(|_| std::fmt::Error)
    }
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        for c in s.chars() {
            self.write_char(c)?;
        }
        Ok(())
    }
}

#[test]
fn test_token_abuse() {
    let mut buf = Vec::new();
    let mut writer = WeirdFormatter::new(&mut buf);
    token_abuse!(
        writer, ("AAA")
        digraph G {
            node[fontname="Helvetica,Arial,sans-serif"; fontsize=9;];
        }
    );

    #[rustfmt::skip]
    assert_eq!(
        buf,
r#"digraph G {
    node [fontname = "Helvetica,Arial,sans-serif"; fontsize = 9;];
}"#.as_bytes()
    );

    // drop(writer);
    // use std::io::Write;
    // std::io::stdout().lock().write(&buf);
}
