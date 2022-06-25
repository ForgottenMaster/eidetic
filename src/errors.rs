//! This module will hold all the errors that the API can emit.

#[cfg(feature = "thiserror")]
use thiserror::Error;

/// This error is emitted when the provided number of elements does not
/// match the expected shape.
#[derive(Debug, Eq, PartialEq)]
#[cfg_attr(feature = "thiserror", derive(Error))]
#[cfg_attr(
    feature = "thiserror",
    error("The provided number of elements does not match the requested shape.")
)]
pub struct ElementCountError;
