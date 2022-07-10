use crate::operations::{forward, initialised, trainable, InitialisedOperation};
use crate::optimisers::base::Optimiser;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::Result;

#[derive(Debug, PartialEq)]
pub struct Operation<T> {
    pub(crate) optimiser: T,
    pub(crate) initialised: initialised::weight_multiply::Operation,
    pub(crate) last_input: Tensor<rank::Two>,
}

impl<T> Sealed for Operation<T> {}
impl<T: Optimiser<Tensor<rank::Two>>> trainable::Operation for Operation<T> {
    type Initialised = initialised::weight_multiply::Operation;

    fn into_initialised(self) -> Self::Initialised {
        self.initialised
    }

    fn init(&mut self, epochs: u16) {
        self.optimiser.init(epochs);
    }

    fn end_epoch(&mut self) {
        self.optimiser.end_epoch();
    }
}

impl<'a, T: 'a + Optimiser<Tensor<rank::Two>>> forward::Forward<'a> for Operation<T> {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type Forward = forward::weight_multiply::Operation<'a, T>;

    fn forward(&'a mut self, input: Self::Input) -> Result<(Self::Forward, Self::Output)> {
        self.last_input = input.clone();
        let output = self.initialised.predict(input)?;
        let forward = forward::weight_multiply::Operation { borrow: self };
        Ok((forward, output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::{Forward, TrainableOperation};
    use crate::optimisers::base::OptimiserFactory;
    use crate::optimisers::NullOptimiser;

    #[test]
    fn test_into_initialised() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((1, 3), [4.0, 5.0, 6.0]).unwrap();
        let operation = Operation {
            optimiser: <NullOptimiser as OptimiserFactory<()>>::instantiate(&NullOptimiser::new()),
            initialised: initialised::weight_multiply::Operation {
                input_neurons: 3,
                parameter: parameter.clone(),
            },
            last_input: Tensor::default(),
        };
        let expected = initialised::weight_multiply::Operation {
            input_neurons: 3,
            parameter,
        };

        // Act
        let output = operation.into_initialised();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_forward_success() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((3, 1), [7.0, 8.0, 9.0]).unwrap();
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let expected = Tensor::<rank::Two>::new((2, 1), [50.0, 122.0]).unwrap();
        let mut operation = Operation {
            optimiser: <NullOptimiser as OptimiserFactory<()>>::instantiate(&NullOptimiser::new()),
            initialised: initialised::weight_multiply::Operation {
                input_neurons: 3,
                parameter,
            },
            last_input: Tensor::default(),
        };

        // Act
        let (_, output) = operation.forward(input).unwrap();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_forward_failure() {
        // Arrange
        let parameter = Tensor::<rank::Two>::new((3, 1), [7.0, 8.0, 9.0]).unwrap();
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut operation = Operation {
            optimiser: <NullOptimiser as OptimiserFactory<()>>::instantiate(&NullOptimiser::new()),
            initialised: initialised::weight_multiply::Operation {
                input_neurons: 2,
                parameter,
            },
            last_input: Tensor::default(),
        };

        // Act
        let result = operation.forward(input);

        // Assert
        assert!(result.is_err());
    }
}
