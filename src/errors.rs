#[cfg(feature = "thiserror")]
use thiserror::Error;

/// This is the error type which is used to report a failure to construct a new
/// tensor from a provided iterator of elements.
#[non_exhaustive]
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "thiserror", derive(Error))]
pub enum TensorConstructionError {
    /// This variant is used in the case that the product of the specified tensor shape components (e.g. rows * columns for a rank 2 tensor) doesn't match the number
    /// of elements provided in the iterator.
    #[cfg_attr(feature="thiserror", error("The provided iterator does not have the correct number of elements. Expected {expected} elements."))]
    InvalidShape {
        /// The number of elements that we expected to get.
        expected: usize,
    },
}
