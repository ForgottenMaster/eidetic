use crate::optimisers;
use crate::optimisers::learning_rate_handlers::LearningRateHandler;
use crate::optimisers::{base, NullOptimiser};
use crate::private::Sealed;
use crate::tensors::rank::Rank;
use crate::tensors::Tensor;
use crate::ElementType;
use ndarray::{azip, Array};

/// This is an implementation of a standard stochastic
/// gradient descent (SGD) optimisation strategy but with
/// an amount of momentum given to the updates.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimiserFactory<T> {
    learning_rate_handler: T,
    momentum: ElementType,
}

impl<T> OptimiserFactory<T> {
    /// Constructs a new instance of the `SGDMomentum` optimiser with the
    /// given learning rate handler to get the learning rate from.
    #[must_use]
    pub const fn new(learning_rate_handler: T, momentum: ElementType) -> Self {
        Self {
            learning_rate_handler,
            momentum,
        }
    }
}

impl<T: LearningRateHandler + Clone, R: Rank> optimisers::base::OptimiserFactory<Tensor<R>>
    for OptimiserFactory<T>
{
    type Optimiser = Optimiser<T, R>;
    fn instantiate(&self) -> Self::Optimiser {
        Self::Optimiser {
            learning_rate_handler: self.learning_rate_handler.clone(),
            momentum: self.momentum,
            velocity: None,
        }
    }
}

impl<T> optimisers::base::OptimiserFactory<()> for OptimiserFactory<T> {
    type Optimiser = optimisers::null::Optimiser;
    fn instantiate(&self) -> Self::Optimiser {
        base::OptimiserFactory::<()>::instantiate(&NullOptimiser::new())
    }
}

pub struct Optimiser<T, R: Rank> {
    learning_rate_handler: T,
    velocity: Option<Array<ElementType, R::Internal>>,
    momentum: ElementType,
}

impl<T, R: Rank> Sealed for Optimiser<T, R> {}
impl<T: LearningRateHandler, R: Rank> optimisers::base::Optimiser<Tensor<R>> for Optimiser<T, R> {
    fn optimise(&mut self, parameter: &mut Tensor<R>, gradient: &Tensor<R>) {
        let (parameter, gradient) = (&mut parameter.0, &gradient.0);
        let velocity = self
            .velocity
            .get_or_insert_with(|| Array::zeros(parameter.raw_dim()));
        let momentum = self.momentum;
        let learning_rate = self.learning_rate_handler.learning_rate();
        azip!((parameter in parameter, gradient in gradient, velocity in velocity) {
            *velocity = (*velocity).mul_add(momentum, gradient * learning_rate);
            *parameter -= *velocity;
        });
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
    use crate::activations::Linear;
    use crate::layers::{Chain, Dense, Input};
    use crate::operations::{
        BackwardOperation, Forward, ForwardOperation, InitialisedOperation, TrainableOperation,
        UninitialisedOperation, WithOptimiser,
    };
    use crate::optimisers::learning_rate_handlers::FixedLearningRateHandler;
    use crate::optimisers::SGDMomentum;
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
        let mut network =
            network.with_optimiser(SGDMomentum::new(FixedLearningRateHandler::new(0.0), 0.0));
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
        let mut network =
            network.with_optimiser(SGDMomentum::new(FixedLearningRateHandler::new(0.001), 0.9));
        let input = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output_gradient = Tensor::<rank::Two>::new((2, 1), [1.0, 2.0]).unwrap();
        network
            .forward(input.clone())
            .unwrap()
            .0
            .backward(output_gradient.clone())
            .unwrap()
            .0
            .optimise();
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
            0.7663690000000001,
            1.7406199999999998,
            2.688492,
            3.65416,
            4.610615,
            5.567699999999999,
            6.922123,
            7.913539999999999,
            8.595231,
            9.48259,
            10.9913,
        ]
        .into_iter();
        #[cfg(feature = "f32")]
        let expected = [
            0.7663690000000001,
            1.7406199,
            2.688492,
            3.65416,
            4.610615,
            5.567699999999999,
            6.922123,
            7.913539999999999,
            8.595231,
            9.48259,
            10.9913,
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
