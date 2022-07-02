use crate::activations::ActivationFunction;
use crate::layers::Layer;
use crate::operations::{initialised, uninitialised, InitialisedOperation, UninitialisedOperation};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Result};

/// This is a "dense" layer which is the most common layer type in
/// a neural network, consisting of a weighted sum of the input with some
/// weights matrix, and a bias term added, and then passed through a specific
/// activation function. This layer is therefore generic over the activation function
/// in use.
pub struct Operation<T> {
    weight_multiply: uninitialised::weight_multiply::Operation,
    bias_add: uninitialised::bias_add::Operation,
    activation_function: T,
}

impl<T: ActivationFunction> Operation<T> {
    /// Constructs a new dense layer with the given output neuron count
    /// and given activation function to use.
    pub const fn new(neurons: u16, activation_function: T) -> Self {
        Self {
            weight_multiply: uninitialised::weight_multiply::Operation::new(neurons),
            bias_add: uninitialised::bias_add::Operation::new(neurons),
            activation_function,
        }
    }
}

impl<T: ActivationFunction> Layer for Operation<T> where
    <T as UninitialisedOperation>::Initialised:
        InitialisedOperation<Input = Tensor<rank::Two>, Output = Tensor<rank::Two>>
{
}
impl<T> Sealed for Operation<T> {}
impl<T: ActivationFunction> UninitialisedOperation for Operation<T>
where
    <T as UninitialisedOperation>::Initialised:
        InitialisedOperation<Input = Tensor<rank::Two>, Output = Tensor<rank::Two>>,
{
    type Initialised = initialised::dense::Operation<T::Initialised>;

    fn with_iter_private(
        self,
        iter: &mut impl Iterator<Item = ElementType>,
        input_neuron_count: u16,
    ) -> Result<(Self::Initialised, u16)> {
        let weight_multiply = self.weight_multiply;
        let weight_multiply = weight_multiply.with_iter_private(iter, input_neuron_count);
        let (weight_multiply, output_neurons) = weight_multiply?;
        let (bias_add, _) = self.bias_add.with_iter_private(iter, input_neuron_count)?;
        let activation_function = self.activation_function;
        let activation_function = activation_function.with_iter_private(iter, output_neurons);
        let activation_function = activation_function?.0;
        let initialised = Self::Initialised {
            weight_multiply,
            bias_add,
            activation_function,
        };
        let tuple = (initialised, output_neurons);
        Ok(tuple)
    }

    fn with_seed_private(self, seed: u64, input_neuron_count: u16) -> (Self::Initialised, u16) {
        let weight_multiply = self.weight_multiply;
        let weight_multiply = weight_multiply.with_seed_private(seed, input_neuron_count);
        let (weight_multiply, output_neurons) = weight_multiply;

        let bias_add = self.bias_add;
        let (bias_add, _) = bias_add.with_seed_private(seed + 1, input_neuron_count);

        let activation_function = self.activation_function;
        let activation_function = activation_function.with_seed_private(seed + 2, output_neurons);
        let (activation_function, _) = activation_function;

        let initialised = Self::Initialised {
            weight_multiply,
            bias_add,
            activation_function,
        };
        (initialised, output_neurons)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::ReLU;

    #[test]
    fn test_with_iter_private_success() {
        // Arrange
        let dense = Operation::new(3, ReLU::new());
        let mut iter = [1.0, 4.0, 2.0, 7.0, 3.0, 1.0].into_iter();
        let expected_weight_multiply_parameter =
            Tensor::<rank::Two>::new((1, 3), [1.0, 4.0, 2.0]).unwrap();
        let expected_bias_add_parameter =
            Tensor::<rank::Two>::new((1, 3), [7.0, 3.0, 1.0]).unwrap();
        let weight_multiply = initialised::weight_multiply::Operation {
            input_neurons: 1,
            parameter: expected_weight_multiply_parameter,
        };
        let bias_add = initialised::bias_add::Operation {
            parameter: expected_bias_add_parameter,
        };
        let activation_function = initialised::relu::Operation {
            neurons: 3,
            factor: 0.0,
        };
        let expected = initialised::dense::Operation {
            weight_multiply,
            bias_add,
            activation_function,
        };

        // Act
        let (dense, output_neurons) = dense.with_iter_private(&mut iter, 1).unwrap();

        // Assert
        assert_eq!(dense, expected);
        assert_eq!(output_neurons, 3);
    }

    #[test]
    fn test_with_iter_private_failure() {
        // Arrange
        let dense = Operation::new(3, ReLU::new());
        let mut iter = [1.0, 4.0, 2.0, 7.0, 3.0].into_iter();

        // Act
        let result = dense.with_iter_private(&mut iter, 1);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_with_seed_private() {
        // Arrange
        let dense = Operation::new(3, ReLU::new());
        let seed = 42;

        #[cfg(feature = "f32")]
        let expected_weight_multiply_parameter =
            Tensor::<rank::Two>::new((1, 3), [-0.8979591, 0.06505203, -0.61546296]).unwrap();
        #[cfg(not(feature = "f32"))]
        let expected_weight_multiply_parameter = Tensor::<rank::Two>::new(
            (1, 3),
            [0.06505210094719227, 0.10465496341600944, 0.3342698606008603],
        )
        .unwrap();

        #[cfg(feature = "f32")]
        let expected_bias_add_parameter =
            Tensor::<rank::Two>::new((1, 3), [1.0385424, 0.61948955, -0.012213111]).unwrap();
        #[cfg(not(feature = "f32"))]
        let expected_bias_add_parameter = Tensor::<rank::Two>::new(
            (1, 3),
            [
                0.6194896314300946,
                -0.19585396452513626,
                -0.25781543623982683,
            ],
        )
        .unwrap();

        let weight_multiply = initialised::weight_multiply::Operation {
            input_neurons: 1,
            parameter: expected_weight_multiply_parameter,
        };
        let bias_add = initialised::bias_add::Operation {
            parameter: expected_bias_add_parameter,
        };
        let activation_function = initialised::relu::Operation {
            neurons: 3,
            factor: 0.0,
        };
        let expected = initialised::dense::Operation {
            weight_multiply,
            bias_add,
            activation_function,
        };

        // Act
        let (dense, output_neurons) = dense.with_seed_private(seed, 1);

        // Assert
        assert_eq!(dense, expected);
        assert_eq!(output_neurons, 3);
    }
}
