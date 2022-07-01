use crate::operations::{backward, forward, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

impl<'a> forward::Construct<'a> for trainable::tanh::Operation {
    type Forward = Forward<'a>;
    fn construct(&'a mut self) -> Self::Forward {
        Forward::<'a>(self)
    }
}

#[derive(Debug, PartialEq)]
pub struct Forward<'a>(&'a mut trainable::tanh::Operation);

impl Sealed for Forward<'_> {}
impl<'a> forward::Operation for Forward<'a> {
    type Output = Tensor<rank::Two>;
    type Input = Tensor<rank::Two>;
    type Backward = backward::tanh::Operation;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        let neurons = output_gradient.0.ncols();
        let expected_neurons = self.0.initialised.neurons as usize;
        if neurons == expected_neurons {
            let partial = self.0.last_output.0.mapv(|elem| 1.0 - elem * elem);
            let input_gradient = Tensor(partial * output_gradient.0);
            Ok((backward::tanh::Operation(()), input_gradient))
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
        let last_output = Tensor::<rank::Two>::new(
            (2, 3),
            [
                0.7615941559557649,
                0.9640275800758169,
                0.9950547536867305,
                0.999329299739067,
                0.9999092042625951,
                0.9999877116507956,
            ],
        )
        .unwrap();
        #[cfg(not(feature = "f32"))]
        let last_output = Tensor::<rank::Two>::new(
            (2, 3),
            [
                0.7615941559557649,
                0.9640275800758169,
                0.9950547536867305,
                0.999329299739067,
                0.9999092042625951,
                0.9999877116507956,
            ],
        )
        .unwrap();
        let mut operation = trainable::tanh::Operation {
            initialised: initialised::tanh::Operation { neurons: 3 },
            last_output,
        };
        let forward = Forward(&mut operation);
        let output_gradient =
            Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        #[cfg(feature = "f32")]
        let expected = Tensor::<rank::Two>::new(
            (2, 3),
            [
                0.41997433,
                0.14130163,
                0.029597998,
                0.0053634644,
                0.00090777874,
                0.00014734268,
            ],
        )
        .unwrap();
        #[cfg(not(feature = "f32"))]
        let expected = Tensor::<rank::Two>::new(
            (2, 3),
            [
                0.41997434161402614,
                0.14130164970632886,
                0.029598111496320634,
                0.005363802732103462,
                0.0009079161547193015,
                0.00014745928443171685,
            ],
        )
        .unwrap();

        // Act
        let input_gradient = forward.backward(output_gradient).unwrap().1;

        // Assert
        assert_eq!(input_gradient, expected);
    }

    #[test]
    fn test_backward_failure() {
        // Arrange
        let mut operation = trainable::tanh::Operation {
            initialised: initialised::tanh::Operation { neurons: 3 },
            last_output: Tensor::default(),
        };
        let forward = Forward(&mut operation);
        let output_gradient = Tensor::<rank::Two>::new((1, 4), [1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let result = forward.backward(output_gradient);

        // Assert
        assert!(result.is_err());
    }
}
