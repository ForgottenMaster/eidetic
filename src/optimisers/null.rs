use crate::optimisers::{Optimiser, OptimiserFactory};
use crate::private::Sealed;
use core::marker::PhantomData;

/// This is an OptimiserFactory which does nothing useful.
/// It does however enable a sensible default for the generic
/// parameter of OperationInitialised when first initialised from
/// an OperationUninitialised.
#[derive(Default)]
pub struct NullOptimiser(());

impl NullOptimiser {
    /// This function will create a new NullOptimiser instance.
    pub fn new() -> Self {
        Self(())
    }
}

impl Sealed for NullOptimiser {}
impl<T> OptimiserFactory<T> for NullOptimiser {
    type Optimiser = NullOptimiserConcrete<T>;
}

pub struct NullOptimiserConcrete<T>(PhantomData<T>);

impl<T> Sealed for NullOptimiserConcrete<T> {}
impl<T> Optimiser for NullOptimiserConcrete<T> {
    type Parameter = T;
}