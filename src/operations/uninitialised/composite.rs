use crate::operations::{initialised, InitialisedOperation, UninitialisedOperation};
use crate::private::Sealed;
use crate::{ElementType, Result};

/// This structure represents a composite, or a chained
/// layer. This is most likely constructed by calling the `.chain(ChainTarget)` method
/// on a `Chain` implementation, which is implemented by the input layer and the composite
/// layer itself.
#[derive(Debug, Eq, PartialEq)]
pub struct Operation<T, U> {
    lhs: T,
    rhs: U,
}

/// This trait provides a method that can be used to chain together two layers into a single
/// composite object. This allows us to build a network by chaining together layers, starting
/// with an input layer.
pub trait Chain: Sealed + Sized {
    /// Function that can be called with a `ChainTarget` and produces a composite operation which
    /// represents the sequence of Self, followed by the provided `ChainTarget`.
    fn chain<T: ChainTarget>(self, rhs: T) -> Operation<Self, T> {
        Operation { lhs: self, rhs }
    }
}

/// This is a marker trait which is used to indicate a viable target for chaining. This is the
/// types that are allowed to appear as the arguments to a chain function call. Whereas the Chain
/// trait specifies what is allowed on the left hand side, this trait indicates what can appear on the
/// right of the chain.
pub trait ChainTarget: Sealed {}

impl<T, U> Sealed for Operation<T, U> {}

impl<T, U> Chain for Operation<T, U> {}

impl<T, U> ChainTarget for Operation<T, U> {}

impl<T: UninitialisedOperation, U: UninitialisedOperation> UninitialisedOperation
    for Operation<T, U>
where
    <U as UninitialisedOperation>::Initialised: InitialisedOperation<
        Input = <<T as UninitialisedOperation>::Initialised as InitialisedOperation>::Output,
    >,
{
    type Initialised = initialised::composite::Operation<
        <T as UninitialisedOperation>::Initialised,
        <U as UninitialisedOperation>::Initialised,
    >;

    fn with_iter_private(
        self,
        iter: &mut impl Iterator<Item = ElementType>,
        input_neuron_count: u16,
    ) -> Result<(Self::Initialised, u16)> {
        let (lhs, input_neuron_count) = self.lhs.with_iter_private(iter, input_neuron_count)?;
        let (rhs, input_neuron_count) = self.rhs.with_iter_private(iter, input_neuron_count)?;
        let initialised = Self::Initialised { lhs, rhs };
        Ok((initialised, input_neuron_count))
    }

    fn with_seed_private(self, seed: u64, input_neuron_count: u16) -> (Self::Initialised, u16) {
        let (lhs, input_neuron_count) = self.lhs.with_seed_private(seed, input_neuron_count);
        let (rhs, input_neuron_count) = self.rhs.with_seed_private(seed + 1, input_neuron_count);
        let initialised = Self::Initialised { lhs, rhs };
        (initialised, input_neuron_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Sigmoid;
    use crate::layers::{Dense, Input};

    #[test]
    fn test_with_iter_private_success() {
        // Arrange
        let composite = Input::new(3).chain(Dense::new(2, Sigmoid::new()));
        let mut iter = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into_iter();
        let expected = initialised::composite::Operation {
            lhs: Input::new(3).with_iter_private(&mut iter, 0).unwrap().0,
            rhs: Dense::new(2, Sigmoid::new())
                .with_iter_private(&mut iter, 3)
                .unwrap()
                .0,
        };
        let mut iter = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into_iter();

        // Act
        let (output, output_neurons) = composite.with_iter_private(&mut iter, 0).unwrap();

        // Assert
        assert_eq!(output, expected);
        assert_eq!(output_neurons, 2);
    }

    #[test]
    fn test_with_iter_private_failure() {
        // Arrange
        let composite = Input::new(3).chain(Dense::new(2, Sigmoid::new()));
        let mut iter = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter();

        // Act
        let result = composite.with_iter_private(&mut iter, 0);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_with_seed_private() {
        // Arrange
        let composite = Input::new(3).chain(Dense::new(2, Sigmoid::new()));
        let expected = initialised::composite::Operation {
            lhs: Input::new(3).with_seed_private(42, 0).0,
            rhs: Dense::new(2, Sigmoid::new()).with_seed_private(43, 3).0,
        };

        // Act
        let (output, output_neurons) = composite.with_seed_private(42, 0);

        // Assert
        assert_eq!(output, expected);
        assert_eq!(output_neurons, 2);
    }
}
