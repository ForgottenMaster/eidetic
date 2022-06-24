//! Module containing the traits and types relating
//! to operations and chains of operations in the trainable typestate.

use crate::operations::initialised;
use crate::optimisers::OptimiserFactory;
use crate::private::Sealed;

/// This trait is implemented on those types that represent
/// an operation that is in a state ready to be trained.
/// This means it has been through the `with_optimiser` function
/// call to bind an optimiser to the network.
pub trait Operation: Sealed {
    /// This is the optimiser type that is in use for this trainable
    /// operation.
    type Factory: OptimiserFactory<
        <Self::Initialised as initialised::Operation<Self::Factory>>::Parameter,
    >;

    /// This is the type of the initialised version of the operation.
    type Initialised: initialised::Operation<Self::Factory>;

    /// Calling this function will "go back" from a trainable
    /// state into an initialised one. This allows the trained network
    /// to be used for inference, or allows a different optimiser to be
    /// used (though the optimiser obviously starts from scratch).
    fn into_initialised(self) -> Self::Initialised;
}
