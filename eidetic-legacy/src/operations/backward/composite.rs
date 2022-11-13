use crate::operations::BackwardOperation;
use crate::private::Sealed;

pub struct Operation<T, U> {
    pub(crate) lhs: T,
    pub(crate) rhs: U,
}

impl<T, U> Sealed for Operation<T, U> {}
impl<T, U> BackwardOperation for Operation<T, U>
where
    T: BackwardOperation,
    U: BackwardOperation,
{
    fn optimise(self) {
        self.lhs.optimise();
        self.rhs.optimise();
    }
}

#[cfg(test)]
mod tests {
    use crate::activations::Sigmoid;
    use crate::layers::{Chain, Dense, Input};
    use crate::operations::{
        BackwardOperation, Forward, ForwardOperation, InitialisedOperation, TrainableOperation,
        UninitialisedOperation, WithOptimiser,
    };
    use crate::optimisers::NullOptimiser;
    use crate::tensors::{rank, Tensor};

    #[test]
    fn test_optimise() {
        // Arrange
        let mut operation = Input::new(3)
            .chain(Dense::new(1, Sigmoid::new()))
            .with_seed(41)
            .with_optimiser(NullOptimiser::new());
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output_gradient = Tensor::<rank::Two>::new((2, 1), [1.0, 1.0]).unwrap();
        let (forward, _) = operation.forward(input).unwrap();
        let (backward, _) = forward.backward(output_gradient).unwrap();
        #[cfg(not(feature = "f32"))]
        let expected = [
            0.06505210094719227,
            0.10465496341600944,
            0.3342698606008603,
            0.6194896314300946,
        ]
        .into_iter();
        #[cfg(feature = "f32")]
        let expected = [-0.8979591, 0.06505203, -0.61546296, 1.0385424].into_iter();

        // Act
        backward.optimise();

        // Assert
        operation
            .into_initialised()
            .iter()
            .zip(expected)
            .for_each(|(output, expected)| {
                assert_eq!(output, expected);
            });
    }
}
