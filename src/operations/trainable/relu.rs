use crate::operations::InitialisedOperation;
use crate::operations::{forward, initialised, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct Operation {
    pub(crate) initialised: initialised::relu::Operation,
    pub(crate) last_output: Tensor<rank::Two>,
}

impl Sealed for Operation {}
impl trainable::Operation for Operation {
    type Initialised = initialised::relu::Operation;

    fn into_initialised(self) -> Self::Initialised {
        self.initialised
    }

    fn init(&mut self, _epochs: u16) {}

    fn end_epoch(&mut self) {}
}

impl<'a> forward::Forward<'a> for Operation {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type Forward = forward::relu::Forward<'a>;

    fn forward(&'a mut self, input: Self::Input) -> Result<(Self::Forward, Self::Output)> {
        self.last_output = self.initialised.predict(input)?;
        let clone = self.last_output.clone();
        Ok((forward::relu::Forward(self), clone))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::{Forward, TrainableOperation};

    #[test]
    fn test_into_initialised() {
        // Arrange
        let operation = Operation {
            initialised: initialised::relu::Operation {
                neurons: 42,
                factor: 0.01,
            },
            last_output: Tensor::default(),
        };
        let expected = initialised::relu::Operation {
            neurons: 42,
            factor: 0.01,
        };

        // Act
        let output = operation.into_initialised();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_forward_success() {
        // Arrange
        let mut operation = Operation {
            initialised: initialised::relu::Operation {
                neurons: 3,
                factor: 0.01,
            },
            last_output: Tensor::default(),
        };
        let input = Tensor::<rank::Two>::new((2, 3), [-1.0, 1.0, 1.0, -2.0, 3.0, -7.0]).unwrap();
        let expected =
            Tensor::<rank::Two>::new((2, 3), [-0.01, 1.0, 1.0, -0.02, 3.0, -0.07]).unwrap();

        // Act
        let (_, output) = operation.forward(input).unwrap();

        // Assert
        assert_eq!(output, expected);
        assert_eq!(operation.last_output, expected);
    }

    #[test]
    fn test_forward_failure() {
        // Arrange
        let mut operation = Operation {
            initialised: initialised::relu::Operation {
                neurons: 2,
                factor: 0.01,
            },
            last_output: Tensor::default(),
        };
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let result = operation.forward(input);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_idempotent_functions() {
        // Arrange
        let mut trainable = Operation {
            initialised: initialised::relu::Operation {
                neurons: 2,
                factor: 0.01,
            },
            last_output: Tensor::default(),
        };
        let expected = trainable.clone();

        // Act
        trainable.init(3);
        trainable.end_epoch();

        // Assert
        assert_eq!(trainable, expected);
    }
}
