use crate::operations::InitialisedOperation;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor, TensorIterator};
use crate::{Error, Result};

pub struct Operation {
    pub(crate) parameter: Tensor<rank::Two>,
}

impl Sealed for Operation {}
impl InitialisedOperation for Operation {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type ParameterIter = TensorIterator<rank::Two>;

    fn iter(&self) -> Self::ParameterIter {
        self.parameter.clone().into_iter()
    }

    fn predict(&self, input: Self::Input) -> Result<Self::Output> {
        if input.0.ncols() == self.parameter.0.ncols() && self.parameter.0.nrows() == 1 {
            Ok(Tensor(input.0 + &self.parameter.0))
        } else {
            Err(Error(()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((1, 5), [1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let operation = Operation { parameter };
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0].into_iter();

        // Act
        let output = operation.iter();

        // Assert
        assert!(output.eq(expected));
    }

    #[test]
    fn test_predict_success() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((1, 5), [1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let operation = Operation { parameter };
        let input = Tensor::<rank::Two>::new(
            (2, 5),
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        )
        .unwrap();
        let expected = Tensor::<rank::Two>::new(
            (2, 5),
            [11.0, 13.0, 15.0, 17.0, 19.0, 16.0, 18.0, 20.0, 22.0, 24.0],
        )
        .unwrap();

        // Act
        let output = operation.predict(input).unwrap();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_predict_failure_incorrect_parameter() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let operation = Operation { parameter };
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let result = operation.predict(input);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_failure_incorrect_input_shape() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        let operation = Operation { parameter };
        let input = Tensor::<rank::Two>::new((2, 2), [1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let result = operation.predict(input);

        // Assert
        assert!(result.is_err());
    }
}
