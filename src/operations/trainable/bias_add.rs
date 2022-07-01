use crate::operations::forward::Construct;
use crate::operations::{initialised, InitialisedOperation, TrainableOperation};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::Result;

#[derive(Debug, PartialEq)]
pub struct Operation<T> {
    pub(crate) optimiser: T,
    pub(crate) initialised: initialised::bias_add::Operation,
    pub(crate) last_input: Tensor<rank::Two>,
}

impl<T> Sealed for Operation<T> {}
impl<T> TrainableOperation for Operation<T> {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type Initialised = initialised::bias_add::Operation;

    fn into_initialised(self) -> Self::Initialised {
        self.initialised
    }

    fn forward<'a>(
        &'a mut self,
        input: Self::Input,
    ) -> Result<(<Self as Construct<'a>>::Forward, Self::Output)>
    where
        Self: Construct<'a>,
    {
        self.last_input = input.clone();
        let output = self.initialised.predict(input)?;
        let forward = self.construct();
        Ok((forward, output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimisers::base::OptimiserFactory;
    use crate::optimisers::NullOptimiser;

    #[test]
    fn test_into_initialised() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        let expected = initialised::bias_add::Operation {
            parameter: parameter.clone(),
        };
        let factory = NullOptimiser::new();
        let last_input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
        let operation = Operation {
            optimiser: <NullOptimiser as OptimiserFactory<Tensor<rank::Two>>>::instantiate(
                &factory,
            ),
            initialised: initialised::bias_add::Operation {
                parameter: parameter.clone(),
            },
            last_input: last_input.clone(),
        };

        // Act
        let initialised = operation.into_initialised();

        // Assert
        assert_eq!(initialised, expected);
    }

    #[test]
    fn test_forward_success() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((1, 5), [1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let mut operation = Operation {
            optimiser: (),
            initialised: initialised::bias_add::Operation { parameter },
            last_input: Tensor::default(),
        };
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
        let (_, output) = operation.forward(input).unwrap();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_forward_failure_incorrect_parameter() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut operation = Operation {
            optimiser: (),
            initialised: initialised::bias_add::Operation { parameter },
            last_input: Tensor::default(),
        };
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Act
        let result = operation.forward(input);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_failure_incorrect_input_shape() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        let mut operation = Operation {
            optimiser: (),
            initialised: initialised::bias_add::Operation { parameter },
            last_input: Tensor::default(),
        };
        let input = Tensor::<rank::Two>::new((2, 2), [1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let result = operation.forward(input);

        // Assert
        assert!(result.is_err());
    }
}
