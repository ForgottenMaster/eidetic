#[cfg(feature = "thiserror")]
use thiserror::Error;

/// The enumeration containing the possible errors that Eidetic can produce.
#[derive(Debug)]
#[cfg_attr(feature = "thiserror", derive(Error))]
pub enum Error {
    /// This variant is used when the construction of a tensor could not be performed successfully due to having
    /// insufficient elements.
    #[cfg_attr(feature = "thiserror", error("Tensor could not be constructed. Requested shape was given as {requested_shape:?}, but there were only {number_of_elements} elements."))]
    TensorConstruction {
        /// The shape that was requested for the rank 4 tensor.
        requested_shape: (usize, usize, usize, usize),

        /// The number of elements that were provided through the iterator given.
        number_of_elements: usize,
    },
}

/// A type alias which allows us to omit the error type when writing framework
/// function signatures in order to ensure everything is using Eidetic's Error
/// enumeration.
pub type Result<T> = core::result::Result<T, Error>;
