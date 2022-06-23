//! Module containing the traits and types relating
//! to operations and chains of operations in the trainable typestate.

use crate::private::Sealed;

/// This trait is implemented on those types that represent
/// an operation that is in a state ready to be trained.
/// This means it has been through the "with_optimiser" function
/// call to bind an optimiser to the network.
pub trait OperationTrainable: Sealed {}

/// This struct is used to bind the trainable state of the underlying
/// operation to a specific optimiser inside an opaque instance.
/// This gives us a wrapping type that can orchestrate the optimisation process.
pub struct Trainable<T, U>(T, U);
