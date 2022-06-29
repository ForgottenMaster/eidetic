//! This submodule contains the traits and types for representing the
//! final stage of an operation in a training epoch. That of the operation
//! having had the backward pass ran and ready for optimisation.

pub mod input;
pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod tanh;

use crate::private::Sealed;

/// This trait represents the state of the operation after having the backward
/// pass applied and is the final state of the operation. At this point if the
/// instance is dropped then it's intended it doesn't update parameters, otherwise
/// it can be applied to optimise the parameters with the calculated gradients.
pub trait Operation: Sealed {
    /// Function which consumes this instance and uses the built in optimiser
    /// to update the parameters of the operation.
    fn optimise(self);
}
