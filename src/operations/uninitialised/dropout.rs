use crate::operations::{initialised, UninitialisedOperation};
use crate::private::Sealed;
use crate::{ElementType, Result};

/// Represents the dropout operation/layer which is the layer that
/// randomly drops out neurons (sets to 0) from the previous layer.
/// When running in inference mode (making predictions), then the weights
/// aren't dropped out but all weights are scaled using the keep probability.
#[derive(Debug, PartialEq)]
pub struct Operation {
    keep_probability: ElementType,
}

impl Operation {
    /// Constructs a new instance of the Dropout layer with the
    /// specified keep probability.
    #[must_use]
    pub const fn new(keep_probability: ElementType) -> Self {
        Self { keep_probability }
    }
}

impl Sealed for Operation {}

impl UninitialisedOperation for Operation {
    type Initialised = initialised::dropout::Operation;

    fn with_iter_private(
        self,
        _iter: &mut impl Iterator<Item = ElementType>,
        input_neuron_count: u16,
    ) -> Result<(Self::Initialised, u16)> {
        let keep_probability = self.keep_probability;
        let seed: Option<u64> = None;
        let initialised = Self::Initialised {
            keep_probability,
            seed,
        };
        Ok((initialised, input_neuron_count))
    }

    fn with_seed_private(self, seed: u64, input_neuron_count: u16) -> (Self::Initialised, u16) {
        let keep_probability = self.keep_probability;
        let seed = Some(seed);
        let initialised = Self::Initialised {
            keep_probability,
            seed,
        };
        (initialised, input_neuron_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        // Arrange
        let keep_probability = 0.8;
        let expected = Operation { keep_probability };

        // Act
        let output = Operation::new(keep_probability);

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_with_iter_private() {
        // Arrange
        let mut iter = [].into_iter();
        let keep_probability = 0.8;
        let input_neuron_count = 3;
        let seed: Option<u64> = None;
        let expected = (
            initialised::dropout::Operation {
                keep_probability,
                seed,
            },
            3,
        );
        let uninitialised = Operation::new(keep_probability);

        // Act
        let output = uninitialised
            .with_iter_private(&mut iter, input_neuron_count)
            .unwrap();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_with_seed_private() {
        // Arrange
        let seed = 42;
        let keep_probability = 0.8;
        let input_neuron_count = 3;
        let expected = (
            initialised::dropout::Operation {
                keep_probability,
                seed: Some(seed),
            },
            3,
        );
        let uninitialised = Operation::new(keep_probability);

        // Act
        let output = uninitialised.with_seed_private(seed, input_neuron_count);

        // Assert
        assert_eq!(output, expected);
    }
}
