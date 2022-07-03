use crate::operations::{initialised, trainable, InitialisedOperation, WithOptimiser};
use crate::optimisers::base::OptimiserFactory;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::Result;
use core::iter::Chain;

#[derive(Debug, PartialEq)]
pub struct Operation<T> {
    pub(crate) weight_multiply: initialised::weight_multiply::Operation,
    pub(crate) bias_add: initialised::bias_add::Operation,
    pub(crate) activation_function: T,
}

impl<T> Sealed for Operation<T> {}
impl<T: InitialisedOperation<Input = Tensor<rank::Two>, Output = Tensor<rank::Two>>>
    InitialisedOperation for Operation<T>
{
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type ParameterIter = Chain<
        Chain<
            <initialised::weight_multiply::Operation as InitialisedOperation>::ParameterIter,
            <initialised::bias_add::Operation as InitialisedOperation>::ParameterIter,
        >,
        <T as InitialisedOperation>::ParameterIter,
    >;

    fn iter(&self) -> Self::ParameterIter {
        let weight_multiply = self.weight_multiply.iter();
        let bias_add = self.bias_add.iter();
        let activation_function = self.activation_function.iter();
        weight_multiply.chain(bias_add).chain(activation_function)
    }

    fn predict(&self, input: Self::Input) -> Result<Self::Output> {
        let input = self.weight_multiply.predict(input)?;
        let input = self.bias_add.predict(input)?;
        let output = self.activation_function.predict(input)?;
        Ok(output)
    }
}

impl<T, U: Clone + OptimiserFactory<Tensor<rank::Two>>> WithOptimiser<U> for Operation<T>
where
    initialised::bias_add::Operation: WithOptimiser<U>,
    T: WithOptimiser<U>,
{
    type Trainable = trainable::dense::Operation<
        <initialised::weight_multiply::Operation as WithOptimiser<U>>::Trainable,
        <initialised::bias_add::Operation as WithOptimiser<U>>::Trainable,
        <T as WithOptimiser<U>>::Trainable,
    >;

    fn with_optimiser(self, factory: U) -> Self::Trainable {
        let weight_multiply = self.weight_multiply.with_optimiser(factory.clone());
        let bias_add = self.bias_add.with_optimiser(factory.clone());
        let activation_function = self.activation_function.with_optimiser(factory);
        Self::Trainable {
            weight_multiply,
            bias_add,
            activation_function,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::ReLU;
    use crate::operations::uninitialised;
    use crate::operations::UninitialisedOperation;
    use crate::optimisers::NullOptimiser;

    #[test]
    fn test_iter() {
        // Arrange
        let weight_multiply = initialised::weight_multiply::Operation {
            input_neurons: 1,
            parameter: Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        };
        let bias_add = initialised::bias_add::Operation {
            parameter: Tensor::<rank::Two>::new((1, 3), [4.0, 7.0, 2.0]).unwrap(),
        };
        let activation_function = initialised::relu::Operation {
            neurons: 3,
            factor: 0.0,
        };
        let dense = Operation {
            weight_multiply,
            bias_add,
            activation_function,
        };
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 7.0, 2.0].into_iter();

        // Act
        let output = dense.iter();

        // Assert
        assert!(output.eq(expected));
    }

    #[test]
    fn test_predict_success() {
        // Arrange
        let weight_multiply = initialised::weight_multiply::Operation {
            input_neurons: 2,
            parameter: Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        };
        let bias_add = initialised::bias_add::Operation {
            parameter: Tensor::<rank::Two>::new((1, 3), [4.0, 7.0, 2.0]).unwrap(),
        };
        let activation_function = initialised::relu::Operation {
            neurons: 3,
            factor: 0.0,
        };
        let dense = Operation {
            weight_multiply,
            bias_add,
            activation_function,
        };
        let input = Tensor::<rank::Two>::new((2, 2), [7.0, 1.0, 2.0, 6.0]).unwrap();
        let expected =
            Tensor::<rank::Two>::new((2, 3), [15.0, 26.0, 29.0, 30.0, 41.0, 44.0]).unwrap();

        // Act
        let output = dense.predict(input).unwrap();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_predict_failure() {
        // Arrange
        let weight_multiply = initialised::weight_multiply::Operation {
            input_neurons: 1,
            parameter: Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        };
        let bias_add = initialised::bias_add::Operation {
            parameter: Tensor::<rank::Two>::new((1, 3), [4.0, 7.0, 2.0]).unwrap(),
        };
        let activation_function = initialised::relu::Operation {
            neurons: 3,
            factor: 0.0,
        };
        let dense = Operation {
            weight_multiply,
            bias_add,
            activation_function,
        };
        let input = Tensor::<rank::Two>::new((2, 2), [7.0, 1.0, 2.0, 6.0]).unwrap();

        // Act
        let result = dense.predict(input);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_with_optimiser() {
        // Arrange
        let activation = ReLU::new();
        let neurons = 3;
        let input_neurons = 2;
        let mut iter = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 7.0, 2.0].into_iter();
        let dense = uninitialised::dense::Operation::new(neurons, activation);
        let (dense, _) = dense.with_iter_private(&mut iter, input_neurons).unwrap();
        let factory = NullOptimiser::new();
        let weight_multiply = uninitialised::weight_multiply::Operation::new(neurons);
        let mut iter = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter();
        let (weight_multiply, _) = weight_multiply
            .with_iter_private(&mut iter, input_neurons)
            .unwrap();
        let weight_multiply = weight_multiply.with_optimiser(factory.clone());
        let bias_add = uninitialised::bias_add::Operation::new(neurons);
        let mut iter = [4.0, 7.0, 2.0].into_iter();
        let (bias_add, _) = bias_add.with_iter_private(&mut iter, neurons).unwrap();
        let bias_add = bias_add.with_optimiser(factory.clone());
        let activation_function = ReLU::new();
        let (activation_function, _) = activation_function
            .with_iter_private(&mut [].into_iter(), neurons)
            .unwrap();
        let activation_function = activation_function.with_optimiser(factory.clone());
        let expected = trainable::dense::Operation {
            weight_multiply,
            bias_add,
            activation_function,
        };

        // Act
        let dense = dense.with_optimiser(factory);

        // Assert
        assert_eq!(dense, expected);
    }
}
