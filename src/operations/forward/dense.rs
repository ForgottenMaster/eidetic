use crate::operations::{backward, ForwardOperation};
use crate::private::Sealed;
use crate::Result;

pub struct Operation<T, U, V> {
    pub(crate) weight_multiply: T,
    pub(crate) bias_add: U,
    pub(crate) activation_function: V,
}

impl<T, U, V> Sealed for Operation<T, U, V> {}
impl<
        T: ForwardOperation<Output = <U as ForwardOperation>::Input>,
        U: ForwardOperation<Output = <V as ForwardOperation>::Input>,
        V: ForwardOperation,
    > ForwardOperation for Operation<T, U, V>
{
    type Output = <V as ForwardOperation>::Output;
    type Input = <T as ForwardOperation>::Input;
    type Backward = backward::dense::Operation<
        <T as ForwardOperation>::Backward,
        <U as ForwardOperation>::Backward,
        <V as ForwardOperation>::Backward,
    >;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        let activation_function = self.activation_function;
        let activation_function_result = activation_function.backward(output_gradient);
        let (activation_function, output_gradient) = activation_function_result?;
        let (bias_add, output_gradient) = self.bias_add.backward(output_gradient)?;
        let (weight_multiply, input_gradient) = self.weight_multiply.backward(output_gradient)?;
        let backward = Self::Backward {
            weight_multiply,
            bias_add,
            activation_function,
        };
        Ok((backward, input_gradient))
    }
}

#[cfg(test)]
mod tests {
    use crate::activations::Sigmoid;
    use crate::layers::Dense;
    use crate::operations::{Forward, ForwardOperation, UninitialisedOperation, WithOptimiser};
    use crate::optimisers::NullOptimiser;
    use crate::tensors::{rank, Tensor};

    #[test]
    fn test_backward_success() {
        // Arrange
        let dense = Dense::new(1, Sigmoid::new());
        let (dense, _) = dense.with_seed_private(42, 3);
        let mut dense = dense.with_optimiser(NullOptimiser::new());
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (forward, _) = dense.forward(input).unwrap();
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
        let dense = Dense::new(1, Sigmoid::new());
        let (dense, _) = dense.with_seed_private(42, 3);
        let mut dense = dense.with_optimiser(NullOptimiser::new());
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (forward, _) = dense.forward(input).unwrap();
        let output_gradient = Tensor::<rank::Two>::new((4, 1), [1.0, 1.0, 1.0, 1.0]).unwrap();

        // Act
        let result = forward.backward(output_gradient);

        // Assert
        assert!(result.is_err());
    }
}
