use crate::operations::{forward, initialised, Forward, ForwardOperation, TrainableOperation};
use crate::private::Sealed;
use crate::Result;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Operation<T, U> {
    pub(crate) lhs: T,
    pub(crate) rhs: U,
}

impl<T, U> Sealed for Operation<T, U> {}
impl<T, U> TrainableOperation for Operation<T, U>
where
    T: TrainableOperation,
    U: TrainableOperation,
{
    type Initialised = initialised::composite::Operation<
        <T as TrainableOperation>::Initialised,
        <U as TrainableOperation>::Initialised,
    >;

    fn into_initialised(self) -> Self::Initialised {
        let lhs = self.lhs.into_initialised();
        let rhs = self.rhs.into_initialised();
        Self::Initialised { lhs, rhs }
    }

    fn init(&mut self, epochs: u16) {
        self.lhs.init(epochs);
        self.rhs.init(epochs);
    }

    fn end_epoch(&mut self) {
        self.lhs.end_epoch();
        self.rhs.end_epoch();
    }
}

impl<'a, T, U> Forward<'a> for Operation<T, U>
where
    T: Forward<'a>,
    U: Forward<'a, Input = <T as Forward<'a>>::Output>,
    <U as Forward<'a>>::Forward:
        ForwardOperation<Input = <<T as Forward<'a>>::Forward as ForwardOperation>::Output>,
{
    type Input = <T as Forward<'a>>::Input;
    type Output = <U as Forward<'a>>::Output;
    type Forward =
        forward::composite::Operation<<T as Forward<'a>>::Forward, <U as Forward<'a>>::Forward>;

    fn forward(&'a mut self, input: Self::Input) -> Result<(Self::Forward, Self::Output)> {
        let (lhs_forward, input) = self.lhs.forward(input)?;
        let (rhs_forward, output) = self.rhs.forward(input)?;
        let forward = Self::Forward {
            lhs: lhs_forward,
            rhs: rhs_forward,
        };
        Ok((forward, output))
    }
}

#[cfg(test)]
mod tests {
    use crate::activations::{ReLU, Sigmoid};
    use crate::layers::{Chain, Dense, Input};
    use crate::operations::{Forward, TrainableOperation, UninitialisedOperation, WithOptimiser};
    use crate::optimisers::NullOptimiser;
    use crate::tensors::{rank, Tensor};

    #[test]
    fn test_into_initialised() {
        // Arrange
        let trainable = Input::new(2)
            .chain(Dense::new(3, ReLU::new()))
            .with_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 7.0, 2.0].into_iter())
            .unwrap()
            .with_optimiser(NullOptimiser::new());
        let expected = Input::new(2)
            .chain(Dense::new(3, ReLU::new()))
            .with_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 7.0, 2.0].into_iter())
            .unwrap();

        // Act
        let output = trainable.into_initialised();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_forward_success() {
        // Arrange
        let mut operation = Input::new(3)
            .chain(Dense::new(1, Sigmoid::new()))
            .with_seed(41)
            .with_optimiser(NullOptimiser::new());
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        #[cfg(not(feature = "f32"))]
        let expected =
            Tensor::<rank::Two>::new((2, 1), [0.8695131771282456, 0.9679719806197726]).unwrap();
        #[cfg(feature = "f32")]
        let expected = Tensor::<rank::Two>::new((2, 1), [0.17140509, 0.0026758423]).unwrap();

        // Act
        let (_, output) = operation.forward(input).unwrap();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_forward_failure() {
        // Arrange
        let mut operation = Input::new(3)
            .chain(Dense::new(1, Sigmoid::new()))
            .with_seed(41)
            .with_optimiser(NullOptimiser::new());
        let input = Tensor::<rank::Two>::new((2, 2), [1.0, 2.0, 3.0, 4.0]).unwrap();

        // Act
        let result = operation.forward(input);

        // Assert
        assert!(result.is_err());
    }
}
