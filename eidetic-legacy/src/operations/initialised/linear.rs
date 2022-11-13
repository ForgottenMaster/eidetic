use crate::operations::trainable;
use crate::operations::{InitialisedOperation, WithOptimiser};
use crate::optimisers::base::OptimiserFactory;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Error, Result};
use core::iter::{empty, Empty};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Operation {
    pub(crate) neurons: u16,
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
            Ok(input)
        } else {
            Err(Error(()))
        }
    }
}

impl<T: OptimiserFactory<()>> WithOptimiser<T> for Operation {
    type Trainable = trainable::linear::Operation;

    fn with_optimiser(self, _optimiser: T) -> Self::Trainable {
        trainable::linear::Operation(self)
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
        let operation = Operation { neurons: 42 };

        // Act
        let iter_count = operation.iter().count();

        // Assert
        assert_eq!(iter_count, 0);
    }

    #[test]
    fn test_predict_success() {
        // Arrange
        let operation = Operation { neurons: 2 };
        let input = Tensor::<rank::Two>::new((2, 2), [1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let output = operation.predict(input.clone()).unwrap();

        // Assert
        assert_eq!(input, output);
    }

    #[test]
    fn test_predict_failure() {
        // Arrange
        let operation = Operation { neurons: 2 };
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let output = operation.predict(input);

        // Assert
        assert!(output.is_err());
    }

    #[test]
    fn test_with_optimiser() {
        // Arrange
        let operation = Operation { neurons: 3 };
        let expected = trainable::linear::Operation(Operation { neurons: 3 });

        // Act
        let output = operation.with_optimiser(NullOptimiser::new());

        // Assert
        assert_eq!(output, expected);
    }
}
