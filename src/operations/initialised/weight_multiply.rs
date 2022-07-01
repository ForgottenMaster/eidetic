use crate::operations::{initialised, trainable, WithOptimiser};
use crate::optimisers::base::OptimiserFactory;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor, TensorIterator};
use crate::{Error, Result};

#[derive(Debug, PartialEq)]
pub struct Operation {
    pub(crate) input_neurons: u16,
    pub(crate) parameter: Tensor<rank::Two>,
}

impl Sealed for Operation {}
impl initialised::Operation for Operation {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type ParameterIter = TensorIterator<rank::Two>;

    fn iter(&self) -> Self::ParameterIter {
        self.parameter.clone().into_iter()
    }

    fn predict(&self, input: Self::Input) -> Result<Self::Output> {
        if input.0.ncols() == self.input_neurons as usize {
            Ok(Tensor(input.0.dot(&self.parameter.0)))
        } else {
            Err(Error(()))
        }
    }
}

impl<T: OptimiserFactory<Tensor<rank::Two>>> WithOptimiser<T> for Operation {
    type Trainable = trainable::weight_multiply::Operation<T::Optimiser>;

    fn with_optimiser(self, optimiser: T) -> Self::Trainable {
        let optimiser = optimiser.instantiate();
        Self::Trainable {
            optimiser,
            initialised: self,
            last_input: Tensor::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::InitialisedOperation;
    use crate::optimisers::NullOptimiser;

    #[test]
    fn test_iter() {
        // Arrange
        let expected = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        let operation = Operation {
            input_neurons: 42,
            parameter: expected.clone(),
        };
        let expected = expected.into_iter();

        // Act
        let output = operation.iter();

        // Assert
        assert!(output.eq(expected));
    }

    #[test]
    fn test_predict_success() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((3, 1), [7.0, 8.0, 9.0]).unwrap();
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let expected = Tensor::<rank::Two>::new((2, 1), [50.0, 122.0]).unwrap();
        let operation = Operation {
            input_neurons: 3,
            parameter,
        };

        // Act
        let output = operation.predict(input).unwrap();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_predict_failure() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((3, 1), [7.0, 8.0, 9.0]).unwrap();
        let input = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        let operation = Operation {
            input_neurons: 1,
            parameter,
        };

        // Act
        let result = operation.predict(input);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_with_optimiser() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((3, 1), [7.0, 8.0, 9.0]).unwrap();
        let operation = Operation {
            input_neurons: 1,
            parameter: parameter.clone(),
        };
        let optimiser =
            <NullOptimiser as OptimiserFactory<f64>>::instantiate(&NullOptimiser::new());
        let expected = trainable::weight_multiply::Operation {
            optimiser,
            initialised: operation,
            last_input: Tensor::default(),
        };
        let operation = Operation {
            input_neurons: 1,
            parameter: parameter,
        };

        // Act
        let output = operation.with_optimiser(NullOptimiser::new());

        // Assert
        assert_eq!(output, expected);
    }
}
