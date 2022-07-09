use crate::operations::{trainable, InitialisedOperation, WithOptimiser};
use crate::optimisers::base::OptimiserFactory;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Result};
use core::iter::{empty, Empty};

#[derive(Clone, Debug, PartialEq)]
pub struct Operation {
    pub(crate) keep_probability: ElementType,
    pub(crate) seed: Option<u64>, // used during forward pass to generate dropout mask
}

impl Sealed for Operation {}

impl InitialisedOperation for Operation {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type ParameterIter = Empty<ElementType>;

    fn iter(&self) -> Self::ParameterIter {
        empty()
    }

    fn predict(&self, input: Self::Input) -> Result<Self::Output> {
        let keep_probability = self.keep_probability;
        let output = Tensor(input.0 * keep_probability);
        Ok(output)
    }
}

impl<T> WithOptimiser<T> for Operation
where
    T: OptimiserFactory<()>,
{
    type Trainable = trainable::dropout::Operation;

    fn with_optimiser(self, _optimiser: T) -> Self::Trainable {
        Self::Trainable { initialised: self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimisers::NullOptimiser;

    #[test]
    fn test_iter() {
        // Arrange
        let expected = [].into_iter();
        let initialised = Operation {
            keep_probability: 0.8,
            seed: None,
        };

        // Act
        let iter = initialised.iter();

        // Assert
        assert!(iter.eq(expected));
    }

    #[test]
    fn test_predict() {
        // Arrange
        let input = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 4.0]).unwrap();
        let initialised = Operation {
            keep_probability: 0.8,
            seed: None,
        };
        let expected = Tensor::<rank::Two>::new((1, 3), [0.8, 1.6, 3.2]).unwrap();

        // Act
        let output = initialised.predict(input).unwrap();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_with_optimiser() {
        // Arrange
        let factory = NullOptimiser::new();
        let initialised = Operation {
            keep_probability: 0.8,
            seed: None,
        };
        let expected = trainable::dropout::Operation {
            initialised: Operation {
                keep_probability: 0.8,
                seed: None,
            },
        };

        // Act
        let output = initialised.with_optimiser(factory);

        // Assert
        assert_eq!(output, expected);
    }
}
