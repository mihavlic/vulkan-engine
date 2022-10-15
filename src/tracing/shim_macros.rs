macro_rules! generate {
    ($dollar:tt, $($name:ident),*) => {
        $(
            #[cfg(feature = "build-tracing")]
            #[macro_export]
            macro_rules! $name {
                ($dollar ($dollar any:tt)*) => {
                    tracing::$name!($dollar ($dollar any)*)
                }
            }

            #[cfg(not(feature = "build-tracing"))]
            #[macro_export]
            macro_rules! $name {
                ($dollar ($dollar any:tt)*) => {}
            }

            pub use $name;
        )+
    }
}

generate! {$, debug, debug_span, enabled, error, error_span, event, event_enabled, info, info_span, metadata, span, span_enabled, trace, trace_span, warn_enabled}

#[cfg(feature = "build-tracing")]
#[macro_export]
macro_rules! warn_ {
  ($($any:tt)*) => {
    tracing::warn!($($any)*)
  }
}
#[cfg(not(feature = "build-tracing"))]
#[macro_export]
macro_rules! warn_ {
    ($($any:tt)*) => {};
}
pub use warn_ as warn;
