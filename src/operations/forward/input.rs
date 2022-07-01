use crate::operations::{backward, forward, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

impl<'a> forward::Construct<'a> for trainable::input::Operation {
    type Forward = Forward<'a>;
    fn construct(&'a mut self) -> Self::Forward {
        Forward::<'a>(self)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Forward<'a>(&'a mut trainable::input::Operation);

impl Sealed for Forward<'_> {}
impl<'a> forward::Operation for Forward<'a> {
    type Output = Tensor<rank::Two>;
    type Input = Tensor<rank::Two>;
    type Backward = backward::input::Operation;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        let neurons = output_gradient.0.ncols();
        let expected_neurons = self.0 .0.neurons as usize;
        if neurons == expected_neurons {
            Ok((backward::input::Operation(()), output_gradient))
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
            trainable::input::Operation(initialised::input::Operation { neurons: 3 });
        let forward = Forward(&mut operation);
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
            trainable::input::Operation(initialised::input::Operation { neurons: 3 });
        let forward = Forward(&mut operation);
        let output_gradient = Tensor::<rank::Two>::new((1, 4), [1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let result = forward.backward(output_gradient);

        // Assert
        assert!(result.is_err());
    }
}
