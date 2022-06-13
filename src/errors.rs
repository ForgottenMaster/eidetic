#[cfg(feature = "thiserror")]
use thiserror::Error;

/// This is the error type which is used to report a failure to construct a new
/// tensor from a provided iterator of elements.
#[non_exhaustive]
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "thiserror", derive(Error))]
pub enum TensorConstructionError {
    /// This variant is used in the case that the specified tensor shape (e.g. rows * columns) doesn't match the number
    /// of elements provided in the iterator.
    #[cfg_attr(feature="thiserror", error("Constructing tensor with an expected {expected} elements, but only provided {actual} elements."))]
    InvalidShape {
        /// The expected total number of elements for the tensor.
        expected: usize,

        /// The actual provided number of elements in the given iterator.
        actual: usize,
    },
}
