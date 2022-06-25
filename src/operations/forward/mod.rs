//! This module will contain all the traits and types for
//! representing an operation that has had the forward pass
//! run on it for training and so will produce a structure
//! ready for running the backward pass.

use crate::private::Sealed;

/// This trait is used to allow us to produce a specific concrete type for a given
/// lifetime for an implementation specific to a trainable operation.
/// This is required because we can't have higher kinded lifetimes (or generic associated types)
/// on the trainable trait and so can't be generic over all lifetimes in a "Forward" associated type.
///
/// Instead then, we implement this trait for our lifetime that matches the borrow on self, and then
/// we can name the correct concrete type in the Output associated type which binds the borrow and
/// the input together.
pub trait Construct<'a>: Sealed {
    /// The type of the input provided to construct the forward
    /// pass stage.
    type Input;

    /// This associated type defines the concrete output type for this forward pass
    /// given the lifetime we were given.
    type Forward;

    #[doc(hidden)]
    fn construct(&'a mut self, input: Self::Input) -> Self::Forward;
}

/// This trait is used to encompass the functionality of an operation that has had
/// the forward pass performed and is ready for a backward pass when given an output
/// gradient.
pub trait Operation: Sealed {
    /// This is the type of the output, but also the type that we expect the
    /// output gradient to be.
    type Output;

    /// This is the type that the input gradient is for this Operation.
    type Input;

    /// This is the type representing this Operation that has had the backward
    /// pass run and is now ready to apply gradients using the optimiser that should
    /// be built in.
    type Backward;

    /// This function maps the forward pass to a backward one, calculating the
    /// gradients ready for optimisation.
    fn backward(self, output_gradient: Self::Output) -> (Self::Backward, Self::Input);
}
