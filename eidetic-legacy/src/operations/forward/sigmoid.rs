use crate::operations::{backward, forward, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

#[derive(Debug, PartialEq)]
pub struct Operation<'a>(pub(crate) &'a mut trainable::sigmoid::Operation);

impl Sealed for Operation<'_> {}
impl<'a> forward::Operation for Operation<'a> {
    type Output = Tensor<rank::Two>;
    type Input = Tensor<rank::Two>;
    type Backward = backward::sigmoid::Operation;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        if output_gradient.0.raw_dim() == self.0.last_output.0.raw_dim() {
            let partial = self.0.last_output.0.mapv(|elem| elem * (1.0 - elem));
            let input_gradient = Tensor(partial * output_gradient.0);
            Ok((backward::sigmoid::Operation(()), input_gradient))
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
        #[cfg(feature = "f32")]
        let last_output = Tensor::<rank::Two>::new((1, 3), [0.002472623, 0.5, 0.9975274]).unwrap();
        #[cfg(not(feature = "f32"))]
        let last_output =
            Tensor::<rank::Two>::new((1, 3), [0.0024726231566347743, 0.5, 0.9975273768433653])
                .unwrap();
        let mut operation = trainable::sigmoid::Operation {
            initialised: initialised::sigmoid::Operation { neurons: 3 },
            last_output,
        };
        let forward = Operation(&mut operation);
        let output_gradient = Tensor::<rank::Two>::new((1, 3), [1.0, 1.0, 1.0]).unwrap();
        #[cfg(feature = "f32")]
        let expected =
            Tensor::<rank::Two>::new((1, 3), [0.0024665091, 0.25, 0.0024664658]).unwrap();
        #[cfg(not(feature = "f32"))]
        let expected =
            Tensor::<rank::Two>::new((1, 3), [0.002466509291360048, 0.25, 0.002466509291359931])
                .unwrap();

        // Act
        let input_gradient = forward.backward(output_gradient).unwrap().1;

        // Assert
        assert_eq!(input_gradient, expected);
    }

    #[test]
    fn test_backward_failure() {
        // Arrange
        let mut operation = trainable::sigmoid::Operation {
            initialised: initialised::sigmoid::Operation { neurons: 3 },
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
