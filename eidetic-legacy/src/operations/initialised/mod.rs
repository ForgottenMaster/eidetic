//! This submodule contains the traits and structures for operations in the
//! initialised state.

pub mod bias_add;
pub mod composite;
pub mod dense;
pub mod dropout;
pub mod input;
pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod tanh;
pub mod weight_multiply;

use crate::private::Sealed;
use crate::{ElementType, Result};

/// This trait is used to represent an operation in an initialised state that has a valid
/// parameter stored internally, and which can be used to run inference or prepared for
/// training by providing an optimiser.
pub trait Operation: Sealed {
    /// The type that is passed into the operation.
    type Input;

    /// The type that is output from the operation.
    type Output;

    /// The iterator type that will be returned when asked for that
    /// iterates over the elements of the (flattened) parameter(s) within
    /// this operation (for example, for serialization/saving after training).
    type ParameterIter: Iterator<Item = ElementType>;

    /// This function can be called to get an iterator over the copies of the elements
    /// stored within this operation's parameter. The parameter is flattened to a single
    /// stream for emitting. This is guaranteed to be the same order as is accepted by the
    /// `with_iter` initialisation function for networks.
    fn iter(&self) -> Self::ParameterIter;

    /// This function can take a given input and run it through the operation/network to produce
    /// the output for it. Can produce an error if (for example) the input is an incorrect shape.
    ///
    /// # Errors
    /// `Error` if the prediction fails such as if the input is incorrectly shaped.
    fn predict(&self, input: Self::Input) -> Result<Self::Output>;
}

/// This trait is used on an Operation type in order to be able to take it
/// to a trainable form. This is generic over the optimiser type.
/// Generic parameter T is the optimiser factory to use
pub trait WithOptimiser<T>: Sealed {
    /// The output type of object that represents the operation in a
    /// trainable state.
    type Trainable;

    /// The function that will take the optimiser and will produce the
    /// trainable typestate.
    fn with_optimiser(self, optimiser: T) -> Self::Trainable;
}
