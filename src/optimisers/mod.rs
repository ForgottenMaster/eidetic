//! This module will contain the traits and structures for the various methods
//! of optimisation that can be used when updating an operation's parameter.

pub mod null;

use crate::private::Sealed;

/// This trait is intended to be implemented for a given optimiser struct
/// for all `OptimiserInitialised` implementations and can then produce a specific
/// optimiser which supports a specific parameter type.
pub trait OptimiserFactory<T>: Sealed {}

/// This trait is implemented for a specific concrete optimiser for
/// a specific parameter type that can be used to optimise an operation with
/// the same parameter type.
pub trait Optimiser: Sealed {}
