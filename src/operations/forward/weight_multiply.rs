use crate::operations::{backward, forward, trainable, ForwardOperation};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

impl<'a, T: 'a> forward::Construct<'a> for trainable::weight_multiply::Operation<T> {
    type Forward = Forward<'a, T>;
    fn construct(&'a mut self) -> Self::Forward {
        Forward { borrow: self }
    }
}

#[repr(C)] // Try to prevent false reports of uncovered lines, when we do in fact use it.
pub struct Forward<'a, T: 'a> {
    borrow: &'a mut trainable::weight_multiply::Operation<T>,
}

impl<'a, T: 'a> Sealed for Forward<'a, T> {}
impl<'a, T: 'a> ForwardOperation for Forward<'a, T> {
    type Output = Tensor<rank::Two>;
    type Input = Tensor<rank::Two>;
    type Backward = backward::weight_multiply::Operation<'a, T>;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        if output_gradient.0.ncols() == self.borrow.initialised.parameter.0.ncols()
            && self.borrow.last_input.0.nrows() == output_gradient.0.nrows()
        {
            let input_gradient = Tensor(
                output_gradient
                    .0
                    .dot(&self.borrow.initialised.parameter.0.clone().reversed_axes()),
            );
            let parameter_gradient = Tensor(
                self.borrow
                    .last_input
                    .0
                    .clone()
                    .reversed_axes()
                    .dot(&output_gradient.0),
            );
            Ok((
                Self::Backward {
                    borrow: self.borrow,
                    parameter_gradient,
                },
                input_gradient,
            ))
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
        let parameter = Tensor::<rank::Two>::new((3, 1), [7.0, 8.0, 9.0]).unwrap();
        let initialised = initialised::weight_multiply::Operation {
            input_neurons: 3,
            parameter,
        };
        let last_input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut train = trainable::weight_multiply::Operation {
            optimiser,
            initialised,
            last_input,
        };
        let output_gradient = Tensor::<rank::Two>::new((2, 1), [1.0, 1.0]).unwrap();
        let expected_input_gradient =
            Tensor::<rank::Two>::new((2, 3), [7.0, 8.0, 9.0, 7.0, 8.0, 9.0]).unwrap();
        let expected_parameter_gradient =
            Tensor::<rank::Two>::new((3, 1), [5.0, 7.0, 9.0]).unwrap();
        let forward = Forward { borrow: &mut train };

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
        let parameter = Tensor::<rank::Two>::new((3, 1), [7.0, 8.0, 9.0]).unwrap();
        let initialised = initialised::weight_multiply::Operation {
            input_neurons: 3,
            parameter,
        };
        let last_input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut train = trainable::weight_multiply::Operation {
            optimiser,
            initialised,
            last_input,
        };
        let output_gradient =
            Tensor::<rank::Two>::new((3, 2), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let forward = Forward { borrow: &mut train };

        // Act
        let result = forward.backward(output_gradient);

        // Assert
        assert!(result.is_err());
    }
}
