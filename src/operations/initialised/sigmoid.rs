use crate::operations::trainable;
use crate::operations::{InitialisedOperation, WithOptimiser};
use crate::optimisers::base::OptimiserFactory;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Error, Result};
use core::iter::{empty, Empty};

#[derive(Debug, Eq, PartialEq)]
pub struct Operation {
    pub(crate) neurons: usize,
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
        if input.0.ncols() == self.neurons {
            Ok(Tensor(input.0.mapv(|elem| 1.0 / (1.0 + (-elem).exp()))))
        } else {
            Err(Error(()))
        }
    }
}

impl<T: OptimiserFactory<()>> WithOptimiser<T> for Operation {
    type Trainable = trainable::sigmoid::Operation;

    fn with_optimiser(self, _optimiser: T) -> Self::Trainable {
        trainable::sigmoid::Operation {
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
        let operation = Operation { neurons: 42 };

        // Act
        let iter_count = operation.iter().count();

        // Assert
        assert_eq!(iter_count, 0);
    }

    #[test]
    fn test_predict_success() {
        // Arrange
        let operation = Operation { neurons: 3 };
        let input = Tensor::<rank::Two>::new((1, 3), [-6.0, 0.0, 6.0]).unwrap();
        #[cfg(feature = "f32")]
        let expected = Tensor::<rank::Two>::new((1, 3), [0.002472623, 0.5, 0.9975274]).unwrap();
        #[cfg(not(feature = "f32"))]
        let expected =
            Tensor::<rank::Two>::new((1, 3), [0.0024726231566347743, 0.5, 0.9975273768433653])
                .unwrap();

        // Act
        let output = operation.predict(input).unwrap();

        // Assert
        assert_eq!(output, expected);
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
        let expected = trainable::sigmoid::Operation {
            initialised: Operation { neurons: 3 },
            last_output: Tensor::default(),
        };

        // Act
        let output = operation.with_optimiser(NullOptimiser::new());

        // Assert
        assert_eq!(output, expected);
    }
}
