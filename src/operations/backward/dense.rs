use crate::operations::BackwardOperation;
use crate::private::Sealed;

#[derive(Debug, Eq, PartialEq)]
pub struct Operation<T, U, V> {
    pub(crate) weight_multiply: T,
    pub(crate) bias_add: U,
    pub(crate) activation_function: V,
}

impl<T, U, V> Sealed for Operation<T, U, V> {}
impl<T: BackwardOperation, U: BackwardOperation, V: BackwardOperation> BackwardOperation
    for Operation<T, U, V>
{
    fn optimise(self) {
        self.weight_multiply.optimise();
        self.bias_add.optimise();
        self.activation_function.optimise();
    }
}

#[cfg(test)]
mod tests {
    use crate::activations::Sigmoid;
    use crate::layers::Dense;
    use crate::operations::{
        BackwardOperation, Forward, ForwardOperation, InitialisedOperation, TrainableOperation,
        UninitialisedOperation, WithOptimiser,
    };
    use crate::optimisers::base::{Optimiser, OptimiserFactory};
    use crate::tensors::{rank, Tensor};

    #[derive(Clone)]
    struct DummyOptimiserFactory;

    impl OptimiserFactory<Tensor<rank::Two>> for DummyOptimiserFactory {
        type Optimiser = DummyOptimiser;

        fn instantiate(&self) -> Self::Optimiser {
            DummyOptimiser
        }
    }

    impl OptimiserFactory<()> for DummyOptimiserFactory {
        type Optimiser = DummyOptimiser;

        fn instantiate(&self) -> Self::Optimiser {
            DummyOptimiser
        }
    }

    struct DummyOptimiser;

    impl Optimiser<Tensor<rank::Two>> for DummyOptimiser {
        fn optimise(&mut self, parameter: &mut Tensor<rank::Two>, gradient: &Tensor<rank::Two>) {
            *parameter = Tensor(&parameter.0 - &gradient.0);
        }
    }

    impl Optimiser<()> for DummyOptimiser {
        fn optimise(&mut self, _parameter: &mut (), _gradient: &()) {}
    }

    #[test]
    fn test_optimise() {
        // Arrange
        let dense = Dense::new(1, Sigmoid::new());
        let (dense, _) = dense.with_seed_private(42, 3);
        let mut dense = dense.with_optimiser(DummyOptimiserFactory);
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (forward, _) = dense.forward(input).unwrap();
        let output_gradient = Tensor::<rank::Two>::new((2, 1), [1.0, 1.0]).unwrap();
        let (backward, _) = forward.backward(output_gradient).unwrap();
        #[cfg(not(feature = "f32"))]
        let expected = [
            -0.17241681240062612,
            -0.27727618721520586,
            -0.19212352731375193,
            0.4750273941466977,
        ]
        .into_iter();
        #[cfg(feature = "f32")]
        let expected = [-1.0506592, -0.23234218, -1.0575513, 0.8938483].into_iter();

        // Act
        backward.optimise();
        let dense = dense.into_initialised();
        let parameters = dense.iter();

        // Assert
        parameters.zip(expected).for_each(|(parameter, expected)| {
            assert_eq!(parameter, expected);
        });
    }

    #[test]
    fn test_dummy_factory_instantiate() {
        // Arrange
        let factory: &dyn OptimiserFactory<(), Optimiser = DummyOptimiser> = &DummyOptimiserFactory;

        // Act
        factory.instantiate();
    }

    #[test]
    fn test_dummy_optimiser_optimise() {
        // Arrange
        let mut parameter = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        let gradient = Tensor::<rank::Two>::new((1, 3), [0.5, 0.5, 0.5]).unwrap();
        let expected = Tensor::<rank::Two>::new((1, 3), [0.5, 1.5, 2.5]).unwrap();
        let mut optimiser = DummyOptimiser;

        // Act
        optimiser.optimise(&mut parameter, &gradient);

        // Assert
        assert_eq!(parameter, expected);
    }

    #[test]
    fn test_dummy_optimiser_optimise_with_unit() {
        // Arrange
        let mut parameter = ();
        let gradient = ();
        let mut optimiser = DummyOptimiser;

        // Act
        optimiser.optimise(&mut parameter, &gradient);
    }
}
