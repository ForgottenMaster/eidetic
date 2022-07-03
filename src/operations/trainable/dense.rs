use crate::operations::{forward, initialised, Forward, TrainableOperation};
use crate::private::Sealed;
use crate::Result;

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

impl<
        'a,
        T: Forward<'a> + TrainableOperation<Initialised = initialised::weight_multiply::Operation>,
        U: Forward<'a, Input = <T as Forward<'a>>::Output>
            + TrainableOperation<Initialised = initialised::bias_add::Operation>,
        V: Forward<'a, Input = <U as Forward<'a>>::Output> + TrainableOperation,
    > Forward<'a> for Operation<T, U, V>
{
    type Input = <T as Forward<'a>>::Input;
    type Output = <V as Forward<'a>>::Output;
    type Forward = forward::dense::Forward<
        <T as Forward<'a>>::Forward,
        <U as Forward<'a>>::Forward,
        <V as Forward<'a>>::Forward,
    >;

    fn forward(&'a mut self, input: Self::Input) -> Result<(Self::Forward, Self::Output)> {
        let (weight_multiply, input) = self.weight_multiply.forward(input)?;
        let (bias_add, input) = self.bias_add.forward(input)?;
        let (activation_function, output) = self.activation_function.forward(input)?;
        let forward = forward::dense::Forward {
            weight_multiply,
            bias_add,
            activation_function,
        };
        Ok((forward, output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Sigmoid;
    use crate::layers::Dense;
    use crate::operations::{trainable, UninitialisedOperation, WithOptimiser};
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

    #[test]
    fn test_forward() {
        // Arrange
        let dense = Dense::new(1, Sigmoid::new());
        let (dense, _) = dense.with_seed_private(42, 3);
        let mut dense = dense.with_optimiser(NullOptimiser::new());
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        #[cfg(not(feature = "f32"))]
        let expected =
            Tensor::<rank::Two>::new((2, 1), [0.8695131771282456, 0.9679719806197726]).unwrap();
        #[cfg(feature = "f32")]
        let expected = Tensor::<rank::Two>::new((2, 1), [0.17140509, 0.0026758423]).unwrap();

        // Act
        let (_, output) = dense.forward(input).unwrap();

        // Assert
        assert_eq!(output, expected);
    }
}
