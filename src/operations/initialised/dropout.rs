use crate::operations::InitialisedOperation;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Result};
use core::iter::{empty, Empty};

#[derive(Debug, PartialEq)]
pub struct Operation {
    pub(crate) keep_probability: ElementType,
}

impl Sealed for Operation {}

impl InitialisedOperation for Operation {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type ParameterIter = Empty<ElementType>;

    fn iter(&self) -> Self::ParameterIter {
        empty()
    }

    fn predict(&self, input: Self::Input) -> Result<Self::Output> {
        let keep_probability = self.keep_probability;
        let output = Tensor(input.0 * keep_probability);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter() {
        // Arrange
        let expected = [].into_iter();
        let initialised = Operation {
            keep_probability: 0.8,
        };

        // Act
        let iter = initialised.iter();

        // Assert
        assert!(iter.eq(expected));
    }

    #[test]
    fn test_predict() {
        // Arrange
        let input = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 4.0]).unwrap();
        let initialised = Operation {
            keep_probability: 0.8,
        };
        let expected = Tensor::<rank::Two>::new((1, 3), [0.8, 1.6, 3.2]).unwrap();

        // Act
        let output = initialised.predict(input).unwrap();

        // Assert
        assert_eq!(output, expected);
    }
}
