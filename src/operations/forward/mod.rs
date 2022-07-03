//! This module will contain all the traits and types for
//! representing an operation that has had the forward pass
//! run on it for training and so will produce a structure
//! ready for running the backward pass.

pub mod bias_add;
pub mod dense;
pub mod input;
pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod tanh;
pub mod weight_multiply;

use crate::operations::TrainableOperation;
use crate::private::Sealed;
use crate::Result;

/// This trait begins a forward pass on an operation and is required to be separate from
/// the `TrainableOperation` trait because we need to vary the `Forward` handle type based on
/// the lifetime of the borrow to self.
pub trait Forward<'a>: Sealed + TrainableOperation {
    /// The type of the input passed into the forward pass.
    type Input;

    /// The type of the output emitted from the forward pass.
    type Output;

    /// This associated type defines the concrete output type for this forward pass
    /// given the lifetime we were given.
    type Forward;

    /// Begins the forward pass of the operation, with the given input.
    ///
    /// # Errors
    /// `Error` if the forward pass can't be performed such as due to the input being incorrectly shaped.
    fn forward(&'a mut self, input: Self::Input) -> Result<(Self::Forward, Self::Output)>;
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
    ///
    /// # Errors
    /// `Error` if the backward pass fails such as due to an invalid shape output gradient.
    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)>;
}
