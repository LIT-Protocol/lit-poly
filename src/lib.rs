//!
//!
#![deny(
    missing_docs,
    missing_debug_implementations,
    unused_qualifications,
    unused_import_braces,
    clippy::unwrap_used
)]
#![warn(
    clippy::cast_precision_loss,
    clippy::checked_conversions,
    clippy::implicit_saturating_sub,
    clippy::mod_module_files,
    clippy::panic,
    clippy::panic_in_result_fn,
    rust_2018_idioms,
    unused_lifetimes
)]

#[macro_use]
mod macros;
mod polynomial;

pub use polynomial::*;
