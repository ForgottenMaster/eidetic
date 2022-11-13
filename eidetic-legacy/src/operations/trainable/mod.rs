//! Module containing the traits and types relating
//! to operations and chains of operations in the trainable typestate.

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

/// This trait is implemented on those types that represent
/// an operation that is in a state ready to be trained.
/// This means it has been through the `with_optimiser` function
/// call to bind an optimiser to the network.
pub trait Operation: Sealed {
    /// This is the type of the initialised version of the operation.
    type Initialised;

    /// Calling this function will "go back" from a trainable
    /// state into an initialised one. This allows the trained network
    /// to be used for inference, or allows a different optimiser to be
    /// used (though the optimiser obviously starts from scratch).
    fn into_initialised(self) -> Self::Initialised;

    /// This function can be called at the beginning of training by the trainer
    /// to initialise the optimisers in the network if needed to account for the
    /// specific number of epochs we'll be training over.
    fn init(&mut self, epochs: u16);

    /// This function can be called at the end of an epoch by the trainer to provide
    /// a chance to update any internal optimisers as needed.
    fn end_epoch(&mut self);
}