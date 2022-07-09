use crate::operations::InitialisedOperation;
use crate::operations::{forward, initialised, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::Result;

#[derive(Debug, PartialEq)]
pub struct Operation {
    pub(crate) initialised: initialised::sigmoid::Operation,
    pub(crate) last_output: Tensor<rank::Two>,
}

impl Sealed for Operation {}
impl trainable::Operation for Operation {
    type Initialised = initialised::sigmoid::Operation;

    fn into_initialised(self) -> Self::Initialised {
        self.initialised
    }

    fn init(&mut self, _epochs: u16) {}

    fn end_epoch(&mut self) {}
}

impl<'a> forward::Forward<'a> for Operation {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type Forward = forward::sigmoid::Forward<'a>;

    fn forward(&'a mut self, input: Self::Input) -> Result<(Self::Forward, Self::Output)> {
        self.last_output = self.initialised.predict(input)?;
        let clone = self.last_output.clone();
        Ok((forward::sigmoid::Forward(self), clone))
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
            initialised: initialised::sigmoid::Operation { neurons: 42 },
            last_output: Tensor::default(),
        };
        let expected = initialised::sigmoid::Operation { neurons: 42 };

        // Act
        let output = operation.into_initialised();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_forward_success() {
        // Arrange
        let mut operation = Operation {
            initialised: initialised::sigmoid::Operation { neurons: 3 },
            last_output: Tensor::default(),
        };
        let input = Tensor::<rank::Two>::new((1, 3), [-6.0, 0.0, 6.0]).unwrap();
        #[cfg(feature = "f32")]
        let expected = Tensor::<rank::Two>::new((1, 3), [0.002472623, 0.5, 0.9975274]).unwrap();
        #[cfg(not(feature = "f32"))]
        let expected =
            Tensor::<rank::Two>::new((1, 3), [0.0024726231566347743, 0.5, 0.9975273768433653])
                .unwrap();

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
            initialised: initialised::sigmoid::Operation { neurons: 2 },
            last_output: Tensor::default(),
        };
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let result = operation.forward(input);

        // Assert
        assert!(result.is_err());
    }
}
