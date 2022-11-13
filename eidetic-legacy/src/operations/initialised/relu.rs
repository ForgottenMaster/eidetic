use crate::operations::trainable;
use crate::operations::{InitialisedOperation, WithOptimiser};
use crate::optimisers::base::OptimiserFactory;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Error, Result};
use core::iter::{empty, Empty};

#[derive(Clone, Debug, PartialEq)]
pub struct Operation {
    pub(crate) neurons: u16,
    pub(crate) factor: ElementType,
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
        if input.0.ncols() == self.neurons as usize {
            Ok(Tensor(input.0.mapv(|elem| {
                if elem < 0.0 {
                    elem * self.factor
                } else {
                    elem
                }
            })))
        } else {
            Err(Error(()))
        }
    }
}

impl<T: OptimiserFactory<()>> WithOptimiser<T> for Operation {
    type Trainable = trainable::relu::Operation;

    fn with_optimiser(self, _optimiser: T) -> Self::Trainable {
        trainable::relu::Operation {
            initialised: self,
            last_output: Tensor::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimisers::NullOptimiser;
    use crate::tensors::*;

    #[test]
    fn test_iter() {
        // Arrange
        let operation = Operation {
            neurons: 42,
            factor: 0.0,
        };

        // Act
        let iter_count = operation.iter().count();

        // Assert
        assert_eq!(iter_count, 0);
    }

    #[test]
    fn test_predict_success() {
        // Arrange
        let operation = Operation {
            neurons: 3,
            factor: 0.01,
        };
        let input = Tensor::<rank::Two>::new((2, 3), [-1.0, 1.0, 1.0, -2.0, 3.0, -7.0]).unwrap();
        let expected =
            Tensor::<rank::Two>::new((2, 3), [-0.01, 1.0, 1.0, -0.02, 3.0, -0.07]).unwrap();

        // Act
        let output = operation.predict(input).unwrap();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_predict_failure() {
        // Arrange
        let operation = Operation {
            neurons: 2,
            factor: 0.0,
        };
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let output = operation.predict(input);

        // Assert
        assert!(output.is_err());
    }

    #[test]
    fn test_with_optimiser() {
        // Arrange
        let operation = Operation {
            neurons: 3,
            factor: 0.01,
        };
        let expected = trainable::relu::Operation {
            initialised: Operation {
                neurons: 3,
                factor: 0.01,
            },
            last_output: Tensor::default(),
        };

        // Act
        let output = operation.with_optimiser(NullOptimiser::new());

        // Assert
        assert_eq!(output, expected);
    }
}
