//! This module contains the optimiser factory/optimiser that does nothing.
//! This is a sensible default `OptimiserFactory` just so we can specify "something"
//! in the generic type of the initialised operation trait as the factory to use
//! in order to turn the initialised operation into a trainable one.

use crate::optimisers;
use crate::private::Sealed;
use core::marker::PhantomData;

/// This is an `OptimiserFactory` which does nothing useful.
/// It does however enable a sensible default for the generic
/// parameter of `OperationInitialised` when first initialised from
/// an `OperationUninitialised`.
#[derive(Debug, Default, PartialEq)]
pub struct OptimiserFactory(());

impl OptimiserFactory {
    /// This function will create a new `NullOptimiser` instance.
    #[must_use]
    pub const fn new() -> Self {
        Self(())
    }
}

impl Sealed for OptimiserFactory {}
impl<T> optimisers::OptimiserFactory<T> for OptimiserFactory {}

/// This struct is the concrete optimiser that is produced by the
/// null `OptimiserFactory`.
pub struct Optimiser<T>(PhantomData<T>);

impl<T> Sealed for Optimiser<T> {}
impl<T> optimisers::Optimiser for Optimiser<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimiser_factory_new() {
        // Arrange/Act
        let factory = OptimiserFactory::new();

        // Assert
        assert_eq!(factory, OptimiserFactory(()));
    }
}
