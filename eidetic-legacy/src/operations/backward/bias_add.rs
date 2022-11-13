use crate::operations::{trainable, BackwardOperation};
use crate::optimisers::base::Optimiser;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};

#[derive(Debug, PartialEq)]
pub struct Operation<'a, T: 'a> {
    pub(crate) borrow: &'a mut trainable::bias_add::Operation<T>,
    pub(crate) parameter_gradient: Tensor<rank::Two>,
}

impl<'a, T: 'a> Sealed for Operation<'a, T> {}
impl<'a, T: Optimiser<Tensor<rank::Two>> + 'a> BackwardOperation for Operation<'a, T> {
    fn optimise(self) {
        let parameter = &mut self.borrow.initialised.parameter;
        let parameter_gradient = &self.parameter_gradient;
        let optimiser = &mut self.borrow.optimiser;
        optimiser.optimise(parameter, parameter_gradient);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::initialised;

    struct DummyOptimiser;

    impl Optimiser<Tensor<rank::Two>> for DummyOptimiser {
        fn optimise(&mut self, parameter: &mut Tensor<rank::Two>, gradient: &Tensor<rank::Two>) {
            *parameter = Tensor(parameter.0.clone() - gradient.0.clone());
        }

        fn init(&mut self, _epochs: u16) {}

        fn end_epoch(&mut self) {}
    }

    #[test]
    fn test_optimise() {
        // Arrange
        let optimiser = DummyOptimiser;
        let parameter = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        let initialised = initialised::bias_add::Operation { parameter };
        let last_input = Tensor::<rank::Two>::new((2, 3), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let mut train = trainable::bias_add::Operation {
            optimiser,
            initialised,
            last_input,
        };
        let parameter_gradient = Tensor::<rank::Two>::new((1, 3), [5.0, 7.0, 9.0]).unwrap();
        let backward = Operation {
            borrow: &mut train,
            parameter_gradient,
        };
        let expected = Tensor::<rank::Two>::new((1, 3), [-4.0, -5.0, -6.0]).unwrap();

        // Act
        backward.optimise();

        // Assert
        assert_eq!(train.initialised.parameter, expected);
    }

    #[test]
    fn test_empty_functions() {
        // Arrange
        let mut optimiser = DummyOptimiser;

        // Act
        optimiser.init(3);
        optimiser.end_epoch();
    }
}
