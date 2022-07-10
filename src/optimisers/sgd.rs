use crate::optimisers;
use crate::optimisers::learning_rate_handlers::LearningRateHandler;
use crate::optimisers::NullOptimiser;
use crate::private::Sealed;
use crate::tensors::rank::Rank;
use crate::tensors::Tensor;

/// This is an implementation of a standard stochastic
/// gradient descent (SGD) optimisation strategy which is
/// simply updating the parameter with some proportion of
/// the gradient.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimiserFactory<T> {
    learning_rate_handler: T,
}

impl<T> OptimiserFactory<T> {
    /// Constructs a new instance of the SGD optimiser with the
    /// given learning rate handler to get the learning rate from.
    #[must_use]
    pub const fn new(learning_rate_handler: T) -> Self {
        Self {
            learning_rate_handler,
        }
    }
}

impl<T: LearningRateHandler + Clone, R: Rank> optimisers::base::OptimiserFactory<Tensor<R>>
    for OptimiserFactory<T>
{
    type Optimiser = Optimiser<T>;
    fn instantiate(&self) -> Self::Optimiser {
        Self::Optimiser {
            learning_rate_handler: self.learning_rate_handler.clone(),
        }
    }
}

impl<T> optimisers::base::OptimiserFactory<()> for OptimiserFactory<T> {
    type Optimiser = <NullOptimiser as optimisers::base::OptimiserFactory<()>>::Optimiser;
    fn instantiate(&self) -> Self::Optimiser {
        <NullOptimiser as optimisers::base::OptimiserFactory<()>>::instantiate(&NullOptimiser::new())
    }
}

pub struct Optimiser<T> {
    learning_rate_handler: T,
}

impl<T> Sealed for Optimiser<T> {}
impl<T: LearningRateHandler, R: Rank> optimisers::base::Optimiser<Tensor<R>> for Optimiser<T> {
    fn optimise(&mut self, parameter: &mut Tensor<R>, gradient: &Tensor<R>) {
        let parameter = &mut parameter.0;
        let gradient = &gradient.0;
        let learning_rate = self.learning_rate_handler.learning_rate();
        *parameter = &*parameter - (gradient * learning_rate);
    }

    fn init(&mut self, epochs: u16) {
        self.learning_rate_handler.init(epochs);
    }

    fn end_epoch(&mut self) {
        self.learning_rate_handler.end_epoch();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Linear;
    use crate::layers::{Chain, Dense, Input};
    use crate::operations::{
        BackwardOperation, Forward, ForwardOperation, InitialisedOperation, TrainableOperation,
        UninitialisedOperation, WithOptimiser,
    };
    use crate::optimisers::base::OptimiserFactory as BaseOptimiserFactory;
    use crate::optimisers::learning_rate_handlers::{
        ExponentialDecayLearningRateHandler, FixedLearningRateHandler,
        LinearDecayLearningRateHandler,
    };
    use crate::optimisers::NullOptimiser;
    use crate::optimisers::SGD;
    use crate::tensors::{rank, Tensor};

    #[test]
    fn test_optimise_idempotent() {
        // Arrange
        let network = Input::new(3)
            .chain(Dense::new(2, Linear::new()))
            .chain(Dense::new(1, Linear::new()));
        let network = network
            .with_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0].into_iter())
            .unwrap();
        let mut network = network.with_optimiser(SGD::new(FixedLearningRateHandler::new(0.0)));
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output_gradient = Tensor::<rank::Two>::new((2, 1), [1.0, 2.0]).unwrap();
        network
            .forward(input)
            .unwrap()
            .0
            .backward(output_gradient)
            .unwrap()
            .0
            .optimise();
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0].into_iter();

        // Act
        let output = network.into_initialised().iter();

        // Assert
        assert!(expected.eq(output));
    }

    #[test]
    fn test_optimise_fixed_rate() {
        // Arrange
        let network = Input::new(3)
            .chain(Dense::new(2, Linear::new()))
            .chain(Dense::new(1, Linear::new()));
        let network = network
            .with_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0].into_iter())
            .unwrap();
        let mut network = network.with_optimiser(SGD::new(FixedLearningRateHandler::new(0.001)));
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output_gradient = Tensor::<rank::Two>::new((2, 1), [1.0, 2.0]).unwrap();
        network
            .forward(input)
            .unwrap()
            .0
            .backward(output_gradient)
            .unwrap()
            .0
            .optimise();
        let expected = [
            0.919, 1.91, 2.892, 3.88, 4.865, 5.85, 6.973, 7.97, 8.859, 9.82, 10.997,
        ]
        .into_iter();

        // Act
        let output = network.into_initialised().iter();

        // Assert
        expected.zip(output).for_each(|(expected, output)| {
            assert_eq!(expected, output);
        });
    }

    #[test]
    fn test_optimise_linear_rate() {
        // Arrange
        let network = Input::new(3)
            .chain(Dense::new(2, Linear::new()))
            .chain(Dense::new(1, Linear::new()));
        let network = network
            .with_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0].into_iter())
            .unwrap();
        let mut network =
            network.with_optimiser(SGD::new(LinearDecayLearningRateHandler::new(0.01, 0.001)));
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output_gradient = Tensor::<rank::Two>::new((2, 1), [1.0, 2.0]).unwrap();
        network.init(3);
        network.end_epoch();
        network
            .forward(input)
            .unwrap()
            .0
            .backward(output_gradient)
            .unwrap()
            .0
            .optimise();
        #[cfg(not(feature = "f32"))]
        let expected = [
            0.5545, 1.505, 2.406, 3.34, 4.2575, 5.175, 6.8515, 7.835, 8.2245, 9.01, 10.9835,
        ]
        .into_iter();
        #[cfg(feature = "f32")]
        let expected = [
            0.5545, 1.505, 2.4060001, 3.3400002, 4.2575, 5.175, 6.8515, 7.835, 8.2245, 9.01,
            10.9835,
        ]
        .into_iter();

        // Act
        let output = network.into_initialised().iter();

        // Assert
        expected.zip(output).for_each(|(expected, output)| {
            assert_eq!(expected, output);
        });
    }

    #[test]
    fn test_optimise_exponential_rate() {
        // Arrange
        let network = Input::new(3)
            .chain(Dense::new(2, Linear::new()))
            .chain(Dense::new(1, Linear::new()));
        let network = network
            .with_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0].into_iter())
            .unwrap();
        let mut network = network.with_optimiser(SGD::new(
            ExponentialDecayLearningRateHandler::new(0.01, 0.001),
        ));
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output_gradient = Tensor::<rank::Two>::new((2, 1), [1.0, 2.0]).unwrap();
        network.init(3);
        network.end_epoch();
        network
            .forward(input)
            .unwrap()
            .0
            .backward(output_gradient)
            .unwrap()
            .0
            .optimise();
        #[cfg(not(feature = "f32"))]
        let expected = [
            0.7438555095263613,
            1.715395010584846,
            2.658474012701815,
            3.6205266807797942,
            4.573092515877269,
            5.5256583509747434,
            6.914618503175454,
            7.905131670194948,
            8.554118849916259,
            9.430790021169692,
            10.990513167019495,
        ]
        .into_iter();
        #[cfg(feature = "f32")]
        let expected = [
            0.7438555, 1.715395, 2.658474, 3.6205268, 4.5730925, 5.525658, 6.9146185, 7.905132,
            8.554119, 9.43079, 10.990513,
        ]
        .into_iter();

        // Act
        let output = network.into_initialised().iter();

        // Assert
        expected.zip(output).for_each(|(expected, output)| {
            assert_eq!(expected, output);
        });
    }

    #[test]
    fn test_instantiate_with_unit() {
        // Arrange
        let optimiser = OptimiserFactory::new(FixedLearningRateHandler::new(0.01));
        let expected =
            <NullOptimiser as BaseOptimiserFactory<()>>::instantiate(&NullOptimiser::new());

        // Act
        let optimiser =
            <OptimiserFactory<FixedLearningRateHandler> as BaseOptimiserFactory<()>>::instantiate(
                &optimiser,
            );

        // Assert
        assert_eq!(optimiser, expected);
    }
}
