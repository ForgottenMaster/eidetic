use crate::operations::forward::Construct;
use crate::operations::{forward, initialised, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

#[derive(Debug, Eq, PartialEq)]
#[repr(C)] // code coverage hack, I dislike <100% in the report :(
pub struct Operation(pub(crate) initialised::linear::Operation);

impl Sealed for Operation {}
impl trainable::Operation for Operation {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type Initialised = initialised::linear::Operation;

    fn into_initialised(self) -> Self::Initialised {
        self.0
    }

    fn forward<'a>(
        &'a mut self,
        input: Self::Input,
    ) -> Result<(<Self as forward::Construct<'a>>::Forward, Self::Output)>
    where
        Self: forward::Construct<'a>,
    {
        if self.0.neurons as usize == input.0.ncols() {
            Ok((self.construct(), input))
        } else {
            Err(Error(()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::TrainableOperation;

    #[test]
    fn test_into_initialised() {
        // Arrange
        let operation = Operation(initialised::linear::Operation { neurons: 42 });
        let expected = initialised::linear::Operation { neurons: 42 };

        // Act
        let output = operation.into_initialised();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_forward_success() {
        // Arrange
        let mut operation = Operation(initialised::linear::Operation { neurons: 2 });
        let input = Tensor::<rank::Two>::new((2, 2), [1.0, 2.0, 3.0, 4.0]).unwrap();
        let expected = input.clone();

        // Act
        let (_, output) = operation.forward(input).unwrap();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_forward_failure() {
        // Arrange
        let mut operation = Operation(initialised::linear::Operation { neurons: 2 });
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let result = operation.forward(input);

        // Assert
        assert!(result.is_err());
    }
}
