use std::{
    ffi::{c_void, CStr},
    fmt::{Arguments, Display},
};

use pumice::vk::{self, DebugUtilsMessageSeverityFlagsEXT};

pub fn to_version(version: u32) -> (u16, u16, u16, u16) {
    (
        vk::api_version_major(version) as u16,
        vk::api_version_minor(version) as u16,
        vk::api_version_patch(version) as u16,
        vk::api_version_variant(version) as u16,
    )
}

pub struct Colored<'a>(tracing::Level, &'a dyn Display);

impl<'a> Display for Colored<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use owo_colors::colors::*;
        match self.0 {
            tracing::Level::TRACE => owo_colors::OwoColorize::fg::<Magenta>(&self.1).fmt(f),
            tracing::Level::INFO => owo_colors::OwoColorize::fg::<Green>(&self.1).fmt(f),
            tracing::Level::DEBUG => owo_colors::OwoColorize::fg::<Blue>(&self.1).fmt(f),
            tracing::Level::WARN => owo_colors::OwoColorize::fg::<Yellow>(&self.1).fmt(f),
            tracing::Level::ERROR => owo_colors::OwoColorize::fg::<Red>(&self.1).fmt(f),
        }
    }
}

pub unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let level = match message_severity {
        DebugUtilsMessageSeverityFlagsEXT::VERBOSE => tracing::Level::TRACE,
        DebugUtilsMessageSeverityFlagsEXT::INFO => tracing::Level::INFO,
        DebugUtilsMessageSeverityFlagsEXT::WARNING => tracing::Level::WARN,
        DebugUtilsMessageSeverityFlagsEXT::ERROR => tracing::Level::ERROR,
        _ => unreachable!(),
    };

    let with_level = |args: Arguments| match level {
        tracing::Level::TRACE => tracing::trace!("{}", args),
        tracing::Level::INFO => tracing::info!("{}", args),
        tracing::Level::WARN => tracing::warn!("{}", args),
        tracing::Level::ERROR => tracing::error!("{}", args),
        _ => unreachable!(),
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
            owo_colors::OwoColorize::fg::<owo_colors::colors::css::Gray>(&msg[i + j + 1..].trim())
        ));
    } else {
        with_level(format_args!("{msg}"));
    }

    vk::FALSE
}
