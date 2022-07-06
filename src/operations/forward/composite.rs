use crate::operations::{backward, ForwardOperation};
use crate::private::Sealed;
use crate::Result;

pub struct Operation<T, U> {
    pub(crate) lhs: T,
    pub(crate) rhs: U,
}

impl<T, U> Sealed for Operation<T, U> {}
impl<T, U> ForwardOperation for Operation<T, U>
where
    T: ForwardOperation<Output = <U as ForwardOperation>::Input>,
    U: ForwardOperation,
{
    type Output = <U as ForwardOperation>::Output;
    type Input = <T as ForwardOperation>::Input;
    type Backward = backward::composite::Operation<
        <T as ForwardOperation>::Backward,
        <U as ForwardOperation>::Backward,
    >;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        let (rhs_backward, output_gradient) = self.rhs.backward(output_gradient)?;
        let (lhs_backward, input_gradient) = self.lhs.backward(output_gradient)?;
        let backward = Self::Backward {
            lhs: lhs_backward,
            rhs: rhs_backward,
        };
        Ok((backward, input_gradient))
    }
}

#[cfg(test)]
mod tests {
    use crate::activations::Sigmoid;
    use crate::layers::{Chain, Dense, Input};
    use crate::operations::{Forward, ForwardOperation, UninitialisedOperation, WithOptimiser};
    use crate::optimisers::NullOptimiser;
    use crate::tensors::{rank, Tensor};

    #[test]
    fn test_backward_success() {
        // Arrange
        let mut operation = Input::new(3)
            .chain(Dense::new(1, Sigmoid::new()))
            .with_seed(41)
            .with_optimiser(NullOptimiser::new());
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (forward, _) = operation.forward(input).unwrap();
        let output_gradient = Tensor::<rank::Two>::new((2, 1), [1.0, 1.0]).unwrap();
        #[cfg(not(feature = "f32"))]
        let expected = Tensor::<rank::Two>::new(
            (2, 3),
            [
                0.007380812149448262,
                0.011874153397566558,
                0.037926262371141654,
                0.0020167598933685184,
                0.0032445367603222224,
                0.01036310954766784,
            ],
        )
        .unwrap();
        #[cfg(feature = "f32")]
        let expected = Tensor::<rank::Two>::new(
            (2, 3),
            [
                -0.127533,
                0.00923904,
                -0.087411374,
                -0.0023963675,
                0.0001736032,
                -0.001642475,
            ],
        )
        .unwrap();

        // Act
        let (_, input_gradient) = forward.backward(output_gradient).unwrap();

        // Assert
        assert_eq!(input_gradient, expected);
    }

    #[test]
    fn test_backward_failure() {
        // Arrange
        let mut operation = Input::new(3)
            .chain(Dense::new(1, Sigmoid::new()))
            .with_seed(41)
            .with_optimiser(NullOptimiser::new());
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (forward, _) = operation.forward(input).unwrap();
        let output_gradient = Tensor::<rank::Two>::new((2, 2), [1.0, 1.0, 1.0, 1.0]).unwrap();

        // Act
        let result = forward.backward(output_gradient);

        // Assert
        assert!(result.is_err());
    }
}
