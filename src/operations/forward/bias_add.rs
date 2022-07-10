use crate::operations::ForwardOperation;
use crate::operations::{backward, trainable};
use crate::optimisers::base::Optimiser;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};
use ndarray::{Array, Axis};

pub struct Operation<'a, T: 'a> {
    pub(crate) borrow: &'a mut trainable::bias_add::Operation<T>,
}

impl<'a, T: 'a> Sealed for Operation<'a, T> {}
impl<'a, T: 'a + Optimiser<Tensor<rank::Two>>> ForwardOperation for Operation<'a, T> {
    type Output = Tensor<rank::Two>;
    type Input = Tensor<rank::Two>;
    type Backward = backward::bias_add::Operation<'a, T>;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        let output_gradient_dim = output_gradient.0.raw_dim();
        let input_dim = self.borrow.last_input.0.raw_dim();
        if output_gradient_dim == input_dim {
            let input_gradient = Tensor(Array::ones(input_dim) * &output_gradient.0);
            let borrow = self.borrow;
            let initialised = &borrow.initialised;
            let parameter = &initialised.parameter.0;
            let parameter_dim = parameter.raw_dim();
            let parameter_gradient = Array::ones(parameter_dim) * output_gradient.0;
            let parameter_cols = parameter_gradient.ncols();
            let parameter_gradient = parameter_gradient
                .map_axis(Axis(0), |view| view.sum())
                .into_shape((1, parameter_cols))
                .unwrap();
            let parameter_gradient = Tensor(parameter_gradient);
            let backward = Self::Backward {
                borrow,
                parameter_gradient,
            };
            let returns = (backward, input_gradient);
            Ok(returns)
        } else {
            Err(Error(()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::initialised;
    use crate::optimisers::base::OptimiserFactory;
    use crate::optimisers::NullOptimiser;

    #[test]
    fn test_backward_success() {
        // Arrange
        let optimiser =
            <NullOptimiser as OptimiserFactory<f64>>::instantiate(&NullOptimiser::new());
        let parameter = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        let initialised = initialised::bias_add::Operation { parameter };
        let last_input = Tensor::<rank::Two>::new((2, 3), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let mut train = trainable::bias_add::Operation {
            optimiser,
            initialised,
            last_input,
        };
        let output_gradient =
            Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let expected_input_gradient =
            Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let expected_parameter_gradient =
            Tensor::<rank::Two>::new((1, 3), [5.0, 7.0, 9.0]).unwrap();
        let forward = Operation { borrow: &mut train };

        // Act
        let (backward, input_gradient) = forward.backward(output_gradient).unwrap();

        // Assert
        assert_eq!(input_gradient, expected_input_gradient);
        assert_eq!(backward.parameter_gradient, expected_parameter_gradient);
    }

    #[test]
    fn test_backward_failure() {
        // Arrange
        let optimiser =
            <NullOptimiser as OptimiserFactory<f64>>::instantiate(&NullOptimiser::new());
        let parameter = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        let initialised = initialised::bias_add::Operation { parameter };
        let last_input = Tensor::<rank::Two>::new((2, 3), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let mut train = trainable::bias_add::Operation {
            optimiser,
            initialised,
            last_input,
        };
        let output_gradient = Tensor::<rank::Two>::new((2, 2), [1.0, 2.0, 3.0, 4.0]).unwrap();
        let forward = Operation { borrow: &mut train };

        // Act
        let result = forward.backward(output_gradient);

        // Assert
        assert!(result.is_err());
    }
}
