#[cfg(feature = "thiserror")]
use thiserror::Error;

/// This error type represents the type of error that occurs where some expected
/// number of elements does not match the actual number provided.
///
/// If the Cargo feature "thiserror" is enabled, this will use it to derive the
/// std::error::Error implementation.
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "thiserror", derive(Error))]
#[cfg_attr(
    feature = "thiserror",
    error("Invalid size error. Expected {expected} elements, got {actual}.")
)]
pub struct InvalidSizeError {
    /// The number of elements that were actually expected.
    pub expected: usize,

    /// The number of elements that were provided.
    pub actual: usize,
}
