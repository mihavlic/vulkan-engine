pub mod shim_macros;
pub mod tracing_subscriber;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Trace,
    Info,
    Debug,
    Warn,
    Error,
}

#[cfg(feature = "build-tracing")]
impl Severity {
    pub fn from_tracing_level(level: tracing::Level) -> Self {
        if level == tracing::Level::TRACE {
            Self::Trace
        } else if level == tracing::Level::INFO {
            Self::Info
        } else if level == tracing::Level::WARN {
            Self::Warn
        } else if level == tracing::Level::ERROR {
            Self::Error
        } else {
            unimplemented!("Unknown level {:?}", level);
        }
    }
    pub fn into_tracing_level(&self) -> tracing::Level {
        match self {
            Severity::Trace => tracing::Level::TRACE,
            Severity::Info => tracing::Level::INFO,
            Severity::Debug => tracing::Level::DEBUG,
            Severity::Warn => tracing::Level::WARN,
            Severity::Error => tracing::Level::ERROR,
        }
    }
}
