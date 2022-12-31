use crate::operations::{backward, forward, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

#[derive(Debug, PartialEq)]
pub struct Operation<'a>(pub(crate) &'a mut trainable::tanh::Operation);

impl Sealed for Operation<'_> {}
impl<'a> forward::Operation for Operation<'a> {
    type Output = Tensor<rank::Two>;
    type Input = Tensor<rank::Two>;
    type Backward = backward::tanh::Operation;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        if output_gradient.0.raw_dim() == self.0.last_output.0.raw_dim() {
            let partial = self.0.last_output.0.mapv(|elem| elem.mul_add(-elem, 1.0));
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
        let forward = Operation(&mut operation);
        let output_gradient =
            Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        #[cfg(feature = "f32")]
        let expected = Tensor::<rank::Two>::new(
            (2, 3),
            [
                0.4199743,
                0.14130163,
                0.029597946,
                0.0053635724,
                0.0009077375,
                0.00014734178,
            ],
        )
        .unwrap();
        #[cfg(not(feature = "f32"))]
        let expected = Tensor::<rank::Two>::new(
            (2, 3),
            [
                0.41997434161402614,
                0.14130164970632886,
                0.0295981114963205,
                0.005363802732103666,
                0.0009079161547192634,
                0.00014745928443181785,
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
        let forward = Operation(&mut operation);
        let output_gradient = Tensor::<rank::Two>::new((1, 4), [1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let result = forward.backward(output_gradient);

        // Assert
        assert!(result.is_err());
    }
}
