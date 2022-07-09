//! This module contains the optimiser factory/optimiser that does nothing.
//! This is a sensible default `OptimiserFactory` just so we can specify "something"
//! in the generic type of the initialised operation trait as the factory to use
//! in order to turn the initialised operation into a trainable one.

use crate::optimisers;
use crate::private::Sealed;

/// This is an optimiser that does nothing during the optimisation
/// step of training. Analagous to the Linear activation function where
/// one needs to provide an optimiser to the API but might not want to
/// necessarily do anything.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OptimiserFactory(());

impl OptimiserFactory {
    /// Constructs a new instance of the null optimiser.
    #[must_use]
    pub const fn new() -> Self {
        Self(())
    }
}

impl Sealed for OptimiserFactory {}
impl<T> optimisers::base::OptimiserFactory<T> for OptimiserFactory {
    type Optimiser = Optimiser;
    fn instantiate(&self) -> Self::Optimiser {
        Optimiser::new()
    }
}

/// This struct is the concrete optimiser that is produced by the
/// null `OptimiserFactory`.
#[derive(Debug, Eq, PartialEq)]
pub struct Optimiser(());

impl Optimiser {
    const fn new() -> Self {
        Self(())
    }
}

impl Sealed for Optimiser {}
impl<T> optimisers::base::Optimiser<T> for Optimiser {
    fn optimise(&mut self, _parameter: &mut T, _gradient: &T) {}
    fn init(&mut self, _epochs: u16) {}
    fn end_epoch(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimisers::base::Optimiser as BaseOptimiser;
    use crate::optimisers::base::OptimiserFactory as BaseOptimiserFactory;
    use crate::tensors::*;

    #[test]
    fn test_instantiate() {
        // Arrange
        let expected = Optimiser::new();
        let factory = OptimiserFactory::new();

        // Act
        let optimiser = BaseOptimiserFactory::<f64>::instantiate(&factory);

        // Assert
        assert_eq!(optimiser, expected);
    }

    #[test]
    fn test_optimise() {
        // Arrange
        let mut optimiser = Optimiser::new();
        let mut parameter = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        let gradient = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();

        // Act
        optimiser.optimise(&mut parameter, &gradient);
    }
}
