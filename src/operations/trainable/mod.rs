//! Module containing the traits and types relating
//! to operations and chains of operations in the trainable typestate.

pub mod bias_add;
pub mod input;
pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod tanh;
pub mod weight_multiply;

use crate::operations::forward;
use crate::private::Sealed;
use crate::Result;

/// This trait is implemented on those types that represent
/// an operation that is in a state ready to be trained.
/// This means it has been through the `with_optimiser` function
/// call to bind an optimiser to the network.
pub trait Operation: Sealed {
    /// The type of the input expected into this operation.
    type Input;

    /// The type of the output produced by the flow of the tensors through
    /// the operation.
    type Output;

    /// This is the type of the initialised version of the operation.
    type Initialised;

    /// Calling this function will "go back" from a trainable
    /// state into an initialised one. This allows the trained network
    /// to be used for inference, or allows a different optimiser to be
    /// used (though the optimiser obviously starts from scratch).
    fn into_initialised(self) -> Self::Initialised;

    /// Function which uses the lifetime of the mutable borrow of self
    /// to produce a type that is specific to this Operation and which
    /// should hold onto that wrapped reference until the pass is applied
    /// or dropped.
    ///
    /// # Errors
    /// `Error` if the forward pass can't be performed such as due to the input being incorrectly shaped.
    fn forward<'a>(&'a mut self, input: Self::Input) -> Result<(Self::Forward, Self::Output)>
    where
        Self: forward::Construct<'a>;
}
