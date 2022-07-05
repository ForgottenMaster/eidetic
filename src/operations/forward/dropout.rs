use crate::operations::{backward, trainable, ForwardOperation};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

#[derive(Debug, PartialEq)]
pub struct Operation<'a> {
    pub(crate) _borrow: &'a mut trainable::dropout::Operation,
    pub(crate) mask: Tensor<rank::Two>,
}

impl<'a> Sealed for Operation<'a> {}

impl<'a> ForwardOperation for Operation<'a> {
    type Output = Tensor<rank::Two>;
    type Input = Tensor<rank::Two>;
    type Backward = backward::dropout::Operation<'a>;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        let output_shape = output_gradient.0.raw_dim();
        let mask_shape = self.mask.0.raw_dim();
        if output_shape == mask_shape {
            let input_gradient = Tensor(output_gradient.0 * &self.mask.0);
            let backward = Self::Backward { _forward: self };
            Ok((backward, input_gradient))
        } else {
            Err(Error(()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::initialised;

    #[test]
    fn test_backward_success() {
        // Arrange
        let mut expected_backing = trainable::dropout::Operation {
            initialised: initialised::dropout::Operation {
                keep_probability: 0.6,
                seed: Some(42),
            },
        };
        let mut working_backing = trainable::dropout::Operation {
            initialised: initialised::dropout::Operation {
                keep_probability: 0.6,
                seed: Some(42),
            },
        };
        let expected_backward = backward::dropout::Operation {
            _forward: Operation {
                _borrow: &mut expected_backing,
                mask: Tensor::<rank::Two>::new((1, 3), [1.0, 0.0, 1.0]).unwrap(),
            },
        };
        let forward = Operation {
            _borrow: &mut working_backing,
            mask: Tensor::<rank::Two>::new((1, 3), [1.0, 0.0, 1.0]).unwrap(),
        };
        let output_gradient = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        let expected_input_gradient = Tensor::<rank::Two>::new((1, 3), [1.0, 0.0, 3.0]).unwrap();

        // Act
        let (backward, input_gradient) = forward.backward(output_gradient).unwrap();

        // Assert
        assert_eq!(backward, expected_backward);
        assert_eq!(input_gradient, expected_input_gradient);
    }

    #[test]
    fn test_backward_failure() {
        // Arrange
        let mut working_backing = trainable::dropout::Operation {
            initialised: initialised::dropout::Operation {
                keep_probability: 0.6,
                seed: Some(42),
            },
        };
        let forward = Operation {
            _borrow: &mut working_backing,
            mask: Tensor::<rank::Two>::new((1, 3), [1.0, 0.0, 1.0]).unwrap(),
        };
        let output_gradient = Tensor::<rank::Two>::new((1, 4), [1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let result = forward.backward(output_gradient);

        // Assert
        assert!(result.is_err());
    }
}
