use super::Severity;

#[cfg(feature = "build-tracing")]
pub fn install_tracing_subscriber(severity: Severity) {
    use crate::util::format_writer::FormatWriter;
    use tracing_subscriber::{
        filter::LevelFilter, prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt,
    };
    let filter = match severity {
        Severity::Trace => LevelFilter::TRACE,
        Severity::Info => LevelFilter::INFO,
        Severity::Debug => LevelFilter::DEBUG,
        Severity::Warn => LevelFilter::WARN,
        Severity::Error => LevelFilter::ERROR,
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(true)
                .with_thread_ids(false)
                .with_target(false)
                .without_time()
                .with_writer(|| FormatWriter::new(std::io::stderr(), "      "))
                .compact(),
        )
        .with(filter)
        .try_init()
        .unwrap_or_else(|_| eprintln!("Failed to set tracing subscriber."));
}

#[cfg(not(feature = "build-tracing"))]
pub fn install_tracing_subscriber(_severity: Severity) {}
