use std::{
    ffi::{c_void, CStr},
    fmt::{Arguments, Display},
};

use pumice::vk::{self, DebugUtilsMessageSeverityFlagsEXT};

use crate::{tracing::Severity, util::format_writer::FormatWriter};

pub fn to_version(version: u32) -> (u16, u16, u16, u16) {
    (
        vk::api_version_major(version) as u16,
        vk::api_version_minor(version) as u16,
        vk::api_version_patch(version) as u16,
        vk::api_version_variant(version) as u16,
    )
}

pub struct Colored<'a>(pub Severity, pub &'a dyn Display);

impl<'a> Display for Colored<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use nu_ansi_term::Color::*;
        let color = match self.0 {
            Severity::Trace => Magenta,
            Severity::Info => Green,
            Severity::Debug => Blue,
            Severity::Warn => Yellow,
            Severity::Error => Red,
        };

        write!(f, "{}{}{}", color.prefix(), self.1, color.suffix())
    }
}

pub unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let level = match message_severity {
        DebugUtilsMessageSeverityFlagsEXT::VERBOSE => Severity::Trace,
        DebugUtilsMessageSeverityFlagsEXT::INFO => Severity::Info,
        DebugUtilsMessageSeverityFlagsEXT::WARNING => Severity::Warn,
        DebugUtilsMessageSeverityFlagsEXT::ERROR => Severity::Error,
        _ => unreachable!(),
    };

    let with_level = |args: Arguments| {
        #[cfg(feature = "build-tracing")]
        match level {
            Severity::Trace => tracing::trace!("{}", args),
            Severity::Info => tracing::info!("{}", args),
            Severity::Warn => tracing::warn!("{}", args),
            Severity::Error => tracing::error!("{}", args),
            _ => unreachable!(),
        };

        #[cfg(not(feature = "build-tracing"))]
        {
            #[rustfmt::skip]
            let text = match level {
                Severity::Trace => "TRACE",
                Severity::Info =>  "INFO ",
                Severity::Debug => "DEBUG",
                Severity::Warn =>  "WARN ",
                Severity::Error => "ERROR",
            };
            let color = Colored(level, &text);
            let mut formatter = FormatWriter::new(std::io::stdout(), "     ");
            use std::io::Write;
            writeln!(formatter, "{color} {args}").unwrap();
        }
    };

    let msg = CStr::from_ptr((*p_callback_data).p_message).to_string_lossy();

    if msg.starts_with("Validation") {
        let k = msg.find("[").unwrap();
        let l = msg.find("]").unwrap();

        let i = msg.find("MessageID").unwrap();
        let j = msg[i..].find('|').unwrap();

        with_level(format_args!(
            "{}: {}\n{}",
            Colored(level, &"Validation"),
            msg[k + 1..l].trim(),
            nu_ansi_term::Color::LightGray.paint(msg[i + j + 1..].trim())
        ));
    } else {
        with_level(format_args!("{msg}"));
    }

    vk::FALSE
}
