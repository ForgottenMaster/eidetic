#[cfg(feature = "thiserror")]
use thiserror::Error;

/// The enumeration that will be used as the single aggregate error type
/// for any errors that might arise from usage of the Eidetic API.
/// By default this is just a simple enumeration and doesn't implement std::error::Error
/// because the library is by default a no_std one.
///
/// In order to implement the error trait correctly, enable the "thiserror" feature of the crate
/// which will add the additional derive macros as needed.
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "thiserror", derive(Error))]
pub enum Error {
    /// The error type that is emitted when trying to create a new Tensor from input data
    /// when the length of the input data doesn't match the requested shape of the Tensor.
    #[cfg_attr(feature = "thiserror", error("Invalid shape specified for creating a new Tensor. Provided shape specifies data has {expected} elements. Received {actual} elements."))]
    TensorTryFromIterError {
        /// The expected length of the data calculated from the requested shape
        /// of the Tensor being constructed.
        expected: usize,

        /// The length of the provided input data.
        actual: usize,
    },
}
