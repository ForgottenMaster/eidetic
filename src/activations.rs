//! This module contains any re-exported operations that are used as
//! activation functions in the layers of the neural network.

pub use crate::operations::uninitialised::linear::Operation as Linear;
pub use crate::operations::uninitialised::sigmoid::Operation as Sigmoid;

/// This marker trait is used to identify those operations that are
/// considered activation functions that can then be used to define a layer.
pub trait ActivationFunction: crate::operations::UninitialisedOperation {}
