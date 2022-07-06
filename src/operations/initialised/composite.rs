use crate::operations::InitialisedOperation;
use crate::private::Sealed;
use crate::Result;
use core::iter::Chain;

#[derive(Debug, PartialEq)]
pub struct Operation<T, U> {
    pub(crate) lhs: T,
    pub(crate) rhs: U,
}

impl<T, U> Sealed for Operation<T, U> {}
impl<
        T: InitialisedOperation,
        U: InitialisedOperation<Input = <T as InitialisedOperation>::Output>,
    > InitialisedOperation for Operation<T, U>
{
    type Input = <T as InitialisedOperation>::Input;
    type Output = <U as InitialisedOperation>::Output;
    type ParameterIter = Chain<
        <T as InitialisedOperation>::ParameterIter,
        <U as InitialisedOperation>::ParameterIter,
    >;

    fn iter(&self) -> Self::ParameterIter {
        let lhs_iter = self.lhs.iter();
        let rhs_iter = self.rhs.iter();
        lhs_iter.chain(rhs_iter)
    }

    fn predict(&self, input: Self::Input) -> Result<Self::Output> {
        let input = self.lhs.predict(input)?;
        let input = self.rhs.predict(input)?;
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use crate::activations::{ReLU, Sigmoid};
    use crate::layers::{Chain, Dense, Input};
    use crate::operations::{InitialisedOperation, UninitialisedOperation};
    use crate::tensors::{rank, Tensor};

    #[test]
    fn test_iter() {
        // Arrange
        let operation = Input::new(3)
            .chain(Dense::new(2, Sigmoid::new()))
            .with_seed(42);
        let input = Input::new(3).with_seed(42);
        let dense = Dense::new(2, Sigmoid::new()).with_seed_private(43, 3).0;
        let expected = input.iter().chain(dense.iter());

        // Act
        let output = operation.iter();

        // Assert
        assert!(output.eq(expected));
    }

    #[test]
    fn test_predict_success() {
        // Arrange
        let operation = Input::new(2)
            .chain(Dense::new(3, ReLU::new()))
            .with_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 7.0, 2.0].into_iter())
            .unwrap();
        let input = Tensor::<rank::Two>::new((2, 2), [7.0, 1.0, 2.0, 6.0]).unwrap();
        let expected =
            Tensor::<rank::Two>::new((2, 3), [15.0, 26.0, 29.0, 30.0, 41.0, 44.0]).unwrap();

        // Act
        let output = operation.predict(input).unwrap();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_predict_failure() {
        // Arrange
        let operation = Input::new(2)
            .chain(Dense::new(3, ReLU::new()))
            .with_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 7.0, 2.0].into_iter())
            .unwrap();
        let input = Tensor::<rank::Two>::new((2, 3), [7.0, 1.0, 2.0, 6.0, 1.0, 2.0]).unwrap();

        // Act
        let result = operation.predict(input);

        // Assert
        assert!(result.is_err());
    }
}
