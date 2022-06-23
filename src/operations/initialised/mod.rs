//! This submodule contains the traits and structures for operations in the
//! initialised state.

use crate::private::Sealed;

/// This trait is used to represent an operation in an initialised state that has a valid
/// parameter stored internally, and which can be used to run inference or prepared for
/// training by providing an optimiser.
pub trait OperationInitialised: Sealed {}
