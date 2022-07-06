use crate::operations::{initialised, TrainableOperation};
use crate::private::Sealed;

#[derive(Debug, PartialEq)]
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
}

#[cfg(test)]
mod tests {
    use crate::activations::ReLU;
    use crate::layers::{Chain, Dense, Input};
    use crate::operations::{TrainableOperation, UninitialisedOperation, WithOptimiser};
    use crate::optimisers::NullOptimiser;

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
}
