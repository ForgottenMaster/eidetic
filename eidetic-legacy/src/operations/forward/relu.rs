use crate::operations::{backward, forward, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

#[derive(Debug, PartialEq)]
pub struct Operation<'a>(pub(crate) &'a mut trainable::relu::Operation);

impl Sealed for Operation<'_> {}
impl<'a> forward::Operation for Operation<'a> {
    type Output = Tensor<rank::Two>;
    type Input = Tensor<rank::Two>;
    type Backward = backward::relu::Operation;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        if output_gradient.0.raw_dim() == self.0.last_output.0.raw_dim() {
            let partial = self.0.last_output.0.mapv(|elem| {
                if elem > 0.0 {
                    1.0
                } else {
                    self.0.initialised.factor
                }
            });
            let input_gradient = Tensor(partial * output_gradient.0);
            Ok((backward::relu::Operation(()), input_gradient))
        } else {
            Err(Error(()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::{initialised, ForwardOperation};

    #[test]
    fn test_backward_success() {
        // Arrange
        let last_output =
            Tensor::<rank::Two>::new((2, 3), [-0.01, 1.0, 1.0, -0.02, 3.0, -0.07]).unwrap();
        let mut operation = trainable::relu::Operation {
            initialised: initialised::relu::Operation {
                neurons: 3,
                factor: 0.01,
            },
            last_output,
        };
        let forward = Operation(&mut operation);
        let output_gradient =
            Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let expected = Tensor::<rank::Two>::new((2, 3), [0.01, 2.0, 3.0, 0.04, 5.0, 0.06]).unwrap();

        // Act
        let input_gradient = forward.backward(output_gradient).unwrap().1;

        // Assert
        assert_eq!(input_gradient, expected);
    }

    #[test]
    fn test_backward_failure() {
        // Arrange
        let mut operation = trainable::relu::Operation {
            initialised: initialised::relu::Operation {
                neurons: 3,
                factor: 0.01,
            },
            last_output: Tensor::default(),
        };
        let forward = Operation(&mut operation);
        let output_gradient = Tensor::<rank::Two>::new((1, 4), [1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let result = forward.backward(output_gradient);

        // Assert
        assert!(result.is_err());
    }
}
