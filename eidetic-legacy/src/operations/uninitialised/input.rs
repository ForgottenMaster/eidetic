use crate::operations::uninitialised::composite::{Chain, ChainTarget};
use crate::operations::{initialised, UninitialisedOperation};
use crate::private::Sealed;
use crate::ElementType;
use crate::Result;

/// This structure represents an input operation which will be used as the very first
/// operation in a sequence. This is to ensure that the neuron count is allowed to be defined
/// for the output if the input layer (number of columns in the output), but that the neuron
/// count is allowed to not be specified for the input. This is different from the Linear activation
/// function for example where the output neuron count is the same as the input - hence they need to be
/// two different functions.
#[derive(Debug, Eq, PartialEq)]
pub struct Operation {
    neuron_count: u16,
}

impl Operation {
    /// This function is used to construct a new Input operation with a given
    /// neuron count. If data is provided when running/training the network and the
    /// neuron/column count doesn't match then this will be an error.
    #[must_use]
    pub const fn new(neuron_count: u16) -> Self {
        Self { neuron_count }
    }
}

impl Sealed for Operation {}
impl Chain for Operation {}
impl ChainTarget for Operation {}
impl UninitialisedOperation for Operation {
    type Initialised = initialised::input::Operation;

    fn with_iter_private(
        self,
        _iter: &mut impl Iterator<Item = ElementType>,
        _input_neuron_count: u16,
    ) -> Result<(Self::Initialised, u16)> {
        Ok((
            initialised::input::Operation {
                neurons: self.neuron_count,
            },
            self.neuron_count,
        ))
    }

    fn with_seed_private(self, _seed: u64, _input_neuron_count: u16) -> (Self::Initialised, u16) {
        (
            initialised::input::Operation {
                neurons: self.neuron_count,
            },
            self.neuron_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        // Arrange
        let expected = Operation { neuron_count: 42 };

        // Act
        let output = Operation::new(42);

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_with_iter() {
        // Arrange
        let operation = Operation::new(122);
        let expected_initialised = initialised::input::Operation { neurons: 122 };

        // Act
        let initialised = operation.with_iter([].into_iter()).unwrap();

        // Assert
        assert_eq!(initialised, expected_initialised);
    }

    #[test]
    fn test_with_seed() {
        // Arrange
        let operation = Operation::new(135);
        let expected_initialised = initialised::input::Operation { neurons: 135 };

        // Act
        let initialised = operation.with_seed(42);

        // Assert
        assert_eq!(initialised, expected_initialised);
    }
}
