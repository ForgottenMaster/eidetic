use crate::operations::{backward, forward, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

#[derive(Debug, Eq, PartialEq)]
pub struct Operation<'a>(pub(crate) &'a mut trainable::linear::Operation);

impl Sealed for Operation<'_> {}
impl<'a> forward::Operation for Operation<'a> {
    type Output = Tensor<rank::Two>;
    type Input = Tensor<rank::Two>;
    type Backward = backward::linear::Operation;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        let neurons = output_gradient.0.ncols();
        let expected_neurons = self.0 .0.neurons as usize;
        if neurons == expected_neurons {
            Ok((backward::linear::Operation(()), output_gradient))
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
        let mut operation =
            trainable::linear::Operation(initialised::linear::Operation { neurons: 3 });
        let forward = Operation(&mut operation);
        let output_gradient = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        let expected = output_gradient.clone();

        // Act
        let input_gradient = forward.backward(output_gradient).unwrap().1;

        // Assert
        assert_eq!(input_gradient, expected);
    }

    #[test]
    fn test_backward_failure() {
        // Arrange
        let mut operation =
            trainable::linear::Operation(initialised::linear::Operation { neurons: 3 });
        let forward = Operation(&mut operation);
        let output_gradient = Tensor::<rank::Two>::new((1, 4), [1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let result = forward.backward(output_gradient);

        // Assert
        assert!(result.is_err());
    }
}
