//! This module will contain all the traits and types for
//! representing an operation that has had the forward pass
//! run on it for training and so will produce a structure
//! ready for running the backward pass.

use crate::operations::{initialised, trainable};
use crate::private::Sealed;

/// This trait is used to allow us to produce a specific concrete type for a given
/// lifetime for an implementation specific to a trainable operation.
/// This is required because we can't have higher kinded lifetimes (or generic associated types)
/// on the trainable trait and so can't be generic over all lifetimes in a "Forward" associated type.
///
/// Instead then, we implement this trait for our lifetime that matches the borrow on self, and then
/// we can name the correct concrete type in the Output associated type which binds the borrow and
/// the input together.
pub trait Construct<'a>: Sealed + trainable::Operation {
    /// This associated type defines the concrete output type for this forward pass
    /// given the lifetime we were given.
    type Output: Operation;

    /// Takes input of the appropriate type and a mutable borrow to self and will
    /// return a specific concrete instance of the appropriate generic type to
    /// represent that forward pass.
    fn construct(
        &'a mut self,
        input: <Self::Initialised as initialised::Operation<Self::Factory>>::Input,
    ) -> Self::Output;
}

/// This trait represents the functionality for an operation in the
/// forward pass state. This will allow the caller to access the final
/// output and also to begin the backpropagation pass given the output
/// gradient.
pub trait Operation: Sealed {}
