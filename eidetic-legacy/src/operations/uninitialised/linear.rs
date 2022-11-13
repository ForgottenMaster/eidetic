use crate::activations::ActivationFunction;
use crate::operations::initialised;
use crate::operations::UninitialisedOperation;
use crate::private::Sealed;
use crate::ElementType;
use crate::Result;

/// This is a linear activation function intended to be used at the end of a dense
/// layer in a neural network. It is linear in that it allows the data to pass through
/// unchanged.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct Operation(());

impl Operation {
    /// This function is used to construct a new Linear activation
    /// to be passed in to a dense layer within a network.
    #[must_use]
    pub const fn new() -> Self {
        Self(())
    }
}

impl Sealed for Operation {}
impl ActivationFunction for Operation {}
impl UninitialisedOperation for Operation {
    type Initialised = initialised::linear::Operation;

    fn with_iter_private(
        self,
        _iter: &mut impl Iterator<Item = ElementType>,
        input_neuron_count: u16,
    ) -> Result<(Self::Initialised, u16)> {
        Ok((
            initialised::linear::Operation {
                neurons: input_neuron_count,
            },
            input_neuron_count,
        ))
    }

    fn with_seed_private(self, _seed: u64, input_neuron_count: u16) -> (Self::Initialised, u16) {
        (
            initialised::linear::Operation {
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
        let expected_initialised = initialised::linear::Operation { neurons: 122 };
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
        let expected_initialised = initialised::linear::Operation { neurons: 135 };

        // Act
        let (initialised, output_neurons) = operation.with_seed_private(42, 135);

        // Assert
        assert_eq!(initialised, expected_initialised);
        assert_eq!(output_neurons, 135);
    }
}
