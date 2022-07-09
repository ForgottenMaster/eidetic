use crate::optimisers;
use crate::optimisers::learning_rate_handlers::LearningRateHandler;
use crate::private::Sealed;
use crate::tensors::rank::Rank;
use crate::tensors::Tensor;
use core::cell::RefCell;

/// This is an implementation of a standard stochastic
/// gradient descent (SGD) optimisation strategy which is
/// simply updating the parameter with some proportion of
/// the gradient.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimiserFactory<'a, T> {
    learning_rate_handler: &'a RefCell<T>,
}

impl<'a, T> OptimiserFactory<'a, T> {
    /// Constructs a new instance of the SGD optimiser with the
    /// given learning rate handler to get the learning rate from.
    #[must_use]
    pub const fn new(learning_rate_handler: &'a RefCell<T>) -> Self {
        Self {
            learning_rate_handler,
        }
    }
}

impl<'a, T: LearningRateHandler, R: Rank> optimisers::base::OptimiserFactory<Tensor<R>>
    for OptimiserFactory<'a, T>
{
    type Optimiser = Optimiser<'a, T>;
    fn instantiate(&self) -> Self::Optimiser {
        Self::Optimiser {
            learning_rate_handler: self.learning_rate_handler,
        }
    }
}

impl<'a, T> optimisers::base::OptimiserFactory<()> for OptimiserFactory<'a, T> {
    type Optimiser = Optimiser<'a, T>;
    fn instantiate(&self) -> Self::Optimiser {
        Self::Optimiser {
            learning_rate_handler: self.learning_rate_handler,
        }
    }
}

pub struct Optimiser<'a, T> {
    learning_rate_handler: &'a RefCell<T>,
}

impl<'a, T> Sealed for Optimiser<'a, T> {}
impl<'a, T: LearningRateHandler, R: Rank> optimisers::base::Optimiser<Tensor<R>>
    for Optimiser<'a, T>
{
    fn optimise(&mut self, parameter: &mut Tensor<R>, gradient: &Tensor<R>) {
        let parameter = &mut parameter.0;
        let gradient = &gradient.0;
        let learning_rate = self.learning_rate_handler.borrow().learning_rate();
        *parameter = &*parameter - (gradient * learning_rate);
    }
}
impl<'a, T> optimisers::base::Optimiser<()> for Optimiser<'a, T> {
    fn optimise(&mut self, _parameter: &mut (), _gradient: &()) {}
}

#[cfg(test)]
mod tests {
    use crate::activations::Linear;
    use crate::layers::{Chain, Dense, Input};
    use crate::operations::{
        BackwardOperation, Forward, ForwardOperation, InitialisedOperation, TrainableOperation,
        UninitialisedOperation, WithOptimiser,
    };
    use crate::optimisers::learning_rate_handlers::{
        ExponentialDecayLearningRateHandler, FixedLearningRateHandler, LearningRateHandler,
        LinearDecayLearningRateHandler,
    };
    use crate::optimisers::SGD;
    use crate::tensors::{rank, Tensor};
    use core::cell::RefCell;

    #[test]
    fn test_optimise_idempotent() {
        // Arrange
        let learning_rate_handler = RefCell::new(FixedLearningRateHandler::new(0.0));
        let network = Input::new(3)
            .chain(Dense::new(2, Linear::new()))
            .chain(Dense::new(1, Linear::new()));
        let network = network
            .with_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0].into_iter())
            .unwrap();
        let mut network = network.with_optimiser(SGD::new(&learning_rate_handler));
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
        let learning_rate_handler = RefCell::new(FixedLearningRateHandler::new(0.001));
        let network = Input::new(3)
            .chain(Dense::new(2, Linear::new()))
            .chain(Dense::new(1, Linear::new()));
        let network = network
            .with_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0].into_iter())
            .unwrap();
        let mut network = network.with_optimiser(SGD::new(&learning_rate_handler));
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
        let learning_rate_handler = RefCell::new(LinearDecayLearningRateHandler::new(0.01, 0.001));
        let network = Input::new(3)
            .chain(Dense::new(2, Linear::new()))
            .chain(Dense::new(1, Linear::new()));
        let network = network
            .with_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0].into_iter())
            .unwrap();
        let mut network = network.with_optimiser(SGD::new(&learning_rate_handler));
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output_gradient = Tensor::<rank::Two>::new((2, 1), [1.0, 2.0]).unwrap();
        learning_rate_handler.borrow_mut().init(3);
        learning_rate_handler.borrow_mut().end_epoch();
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
        let learning_rate_handler =
            RefCell::new(ExponentialDecayLearningRateHandler::new(0.01, 0.001));
        let network = Input::new(3)
            .chain(Dense::new(2, Linear::new()))
            .chain(Dense::new(1, Linear::new()));
        let network = network
            .with_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0].into_iter())
            .unwrap();
        let mut network = network.with_optimiser(SGD::new(&learning_rate_handler));
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output_gradient = Tensor::<rank::Two>::new((2, 1), [1.0, 2.0]).unwrap();
        learning_rate_handler.borrow_mut().init(3);
        learning_rate_handler.borrow_mut().end_epoch();
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
}
