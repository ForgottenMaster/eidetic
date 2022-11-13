//! This module contains any re-exported operations that are used as
//! activation functions in the layers of the neural network.

pub use crate::operations::uninitialised::linear::Operation as Linear;
pub use crate::operations::uninitialised::relu::Operation as ReLU;
pub use crate::operations::uninitialised::sigmoid::Operation as Sigmoid;
pub use crate::operations::uninitialised::tanh::Operation as Tanh;

/// This marker trait is used to identify those operations that are
/// considered activation functions that can then be used to define a layer.
pub trait ActivationFunction: crate::operations::UninitialisedOperation {}
