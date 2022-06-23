//! This module will contain the traits and structures for the various methods
//! of optimisation that can be used when updating an operation's parameter.

use crate::operations::initialised::OperationInitialised;
use crate::private::Sealed;

/// This trait is intended to be implemented for a given optimiser struct
/// for all OptimiserInitialised implementations and can then produce a specific
/// optimiser which supports a specific parameter type.
pub trait OptimiserFactory<T: OperationInitialised>: Sealed {
    /// The type of the optimiser produced by this specific trait implementation
    /// on the factory. Must implement the Optimiser trait and optimise for the same
    /// parameter type as the initialised operation itself.
    type Optimiser: Optimiser<Parameter = T::Parameter>;
}

/// This trait is implemented for a specific concrete optimiser for
/// a specific parameter type that can be used to optimise an operation with
/// the same parameter type.
pub trait Optimiser: Sealed {
    /// This is the type of the parameter/parameter gradient being optimised.
    type Parameter;
}
