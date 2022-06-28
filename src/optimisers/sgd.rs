use crate::optimisers;
use crate::private::Sealed;
use crate::ElementType;
use core::marker::PhantomData;

/// This is an implementation of a standard stochastic
/// gradient descent (SGD) optimisation strategy which is
/// simply updating the parameter with some proportion of
/// the gradient.
#[derive(Debug, PartialEq)]
pub struct OptimiserFactory {
    learning_rate: ElementType,
}

impl OptimiserFactory {
    /// Constructs a new instance of the SGD optimiser with the
    /// given learning rate.
    pub fn new(learning_rate: ElementType) -> Self {
        Self { learning_rate }
    }
}

impl<T> optimisers::base::OptimiserFactory<T> for OptimiserFactory {}

pub struct Optimiser<T>(PhantomData<T>);

impl<T> Sealed for Optimiser<T> {}
impl<T> optimisers::base::Optimiser for Optimiser<T> {}
