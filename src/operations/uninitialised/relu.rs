use crate::activations::ActivationFunction;
use crate::operations::initialised;
use crate::operations::UninitialisedOperation;
use crate::private::Sealed;
use crate::ElementType;
use crate::Result;

/// This is an implementation of the relu activation function
/// which can either be run in leaky mode (negative values
/// are multiplied by a factor) or standard (negative values are mapped
/// to 0).
#[derive(Debug, Default, PartialEq)]
pub struct Operation {
    pub(crate) factor: ElementType,
}

impl Operation {
    /// This function is used to construct a new standard relu
    /// for use in a neural network (standard relu has a factor of 0).
    #[must_use]
    pub const fn new() -> Self {
        Self { factor: 0.0 }
    }

    /// This function is used to construct a "leaky" relu activation function
    /// which only multiplies negative values by some factor.
    #[must_use]
    pub const fn leaky(factor: ElementType) -> Self {
        Self { factor }
    }
}

impl Sealed for Operation {}
impl ActivationFunction for Operation {}
impl UninitialisedOperation for Operation {
    type Initialised = initialised::relu::Operation;

    fn with_iter_private(
        self,
        _iter: &mut impl Iterator<Item = ElementType>,
        input_neuron_count: usize,
    ) -> Result<(Self::Initialised, usize)> {
        Ok((
            initialised::relu::Operation {
                neurons: input_neuron_count,
                factor: self.factor,
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
            initialised::relu::Operation {
                neurons: input_neuron_count,
                factor: self.factor,
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
        let expected = Operation { factor: 0.0 };

        // Act
        let output = Operation::new();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_leaky() {
        // Arrange
        let expected = Operation { factor: 0.01 };

        // Act
        let output = Operation::leaky(0.01);

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_with_iter() {
        // Arrange
        let operation = Operation::new();
        let expected_initialised = initialised::relu::Operation {
            neurons: 122,
            factor: 0.0,
        };
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
        let operation = Operation::leaky(0.01);
        let expected_initialised = initialised::relu::Operation {
            neurons: 135,
            factor: 0.01,
        };

        // Act
        let (initialised, output_neurons) = operation.with_seed_private(42, 135);

        // Assert
        assert_eq!(initialised, expected_initialised);
        assert_eq!(output_neurons, 135);
    }
}
