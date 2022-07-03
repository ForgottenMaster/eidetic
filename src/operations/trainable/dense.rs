use crate::operations::{initialised, TrainableOperation};
use crate::private::Sealed;

#[derive(Debug, PartialEq)]
pub struct Operation<T, U, V> {
    pub(crate) weight_multiply: T,
    pub(crate) bias_add: U,
    pub(crate) activation_function: V,
}

impl<T, U, V> Sealed for Operation<T, U, V> {}
impl<
        T: TrainableOperation<Initialised = initialised::weight_multiply::Operation>,
        U: TrainableOperation<Initialised = initialised::bias_add::Operation>,
        V: TrainableOperation,
    > TrainableOperation for Operation<T, U, V>
{
    type Initialised = initialised::dense::Operation<V::Initialised>;

    fn into_initialised(self) -> Self::Initialised {
        let weight_multiply = self.weight_multiply.into_initialised();
        let bias_add = self.bias_add.into_initialised();
        let activation_function = self.activation_function.into_initialised();
        initialised::dense::Operation {
            weight_multiply,
            bias_add,
            activation_function,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::trainable;
    use crate::optimisers::base::OptimiserFactory;
    use crate::optimisers::NullOptimiser;
    use crate::tensors::{rank, Tensor};

    #[test]
    fn test_into_initialised() {
        // Arrange
        let factory = &NullOptimiser::new();
        let expected = initialised::dense::Operation {
            weight_multiply: initialised::weight_multiply::Operation {
                input_neurons: 1,
                parameter: Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap(),
            },
            bias_add: initialised::bias_add::Operation {
                parameter: Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap(),
            },
            activation_function: initialised::sigmoid::Operation { neurons: 3 },
        };
        let trainable = Operation {
            weight_multiply: trainable::weight_multiply::Operation {
                initialised: initialised::weight_multiply::Operation {
                    input_neurons: 1,
                    parameter: Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap(),
                },
                optimiser: <NullOptimiser as OptimiserFactory<f64>>::instantiate(&factory),
                last_input: Tensor::default(),
            },
            bias_add: trainable::bias_add::Operation {
                initialised: initialised::bias_add::Operation {
                    parameter: Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap(),
                },
                optimiser: <NullOptimiser as OptimiserFactory<f64>>::instantiate(&factory),
                last_input: Tensor::default(),
            },
            activation_function: trainable::sigmoid::Operation {
                initialised: initialised::sigmoid::Operation { neurons: 3 },
                last_output: Tensor::default(),
            },
        };

        // Act
        let output = trainable.into_initialised();

        // Assert
        assert_eq!(output, expected);
    }
}
