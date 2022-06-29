use crate::activations::ActivationFunction;
use crate::operations::initialised;
use crate::operations::UninitialisedOperation;
use crate::private::Sealed;
use crate::ElementType;
use crate::Result;

/// This is an implementation of the tanh nonlinear
/// activation function.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct Operation(());

impl Operation {
    /// This function is used to construct a new Tanh activation
    /// to be passed in to a dense layer within a network.
    #[must_use]
    pub const fn new() -> Self {
        Self(())
    }
}

impl Sealed for Operation {}
impl ActivationFunction for Operation {}
impl UninitialisedOperation for Operation {
    type Initialised = initialised::tanh::Operation;

    fn with_iter_private(
        self,
        _iter: &mut impl Iterator<Item = ElementType>,
        input_neuron_count: usize,
    ) -> Result<(Self::Initialised, usize)> {
        Ok((
            initialised::tanh::Operation {
                neurons: input_neuron_count,
            },
            input_neuron_count,
        ))
    }

    fn with_seed_private(
        self,
        _seed: u64,
        input_neuron_count: usize,
    ) -> (Self::Initialised, usize) {
        (
            initialised::tanh::Operation {
                neurons: input_neuron_count,
            },
            input_neuron_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        // Arrange
        let expected = Operation(());

        // Act
        let output = Operation::new();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_with_iter() {
        // Arrange
        let operation = Operation::new();
        let expected_initialised = initialised::tanh::Operation { neurons: 122 };
        let mut iter = [].into_iter();

        // Act
        let (initialised, output_neurons) = operation.with_iter_private(&mut iter, 122).unwrap();

        // Assert
        assert_eq!(initialised, expected_initialised);
        assert_eq!(output_neurons, 122);
    }

    #[test]
    fn test_with_seed() {
        // Arrange
        let operation = Operation::new();
        let expected_initialised = initialised::tanh::Operation { neurons: 135 };

        // Act
        let (initialised, output_neurons) = operation.with_seed_private(42, 135);

        // Assert
        assert_eq!(initialised, expected_initialised);
        assert_eq!(output_neurons, 135);
    }
}
