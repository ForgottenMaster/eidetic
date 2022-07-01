use crate::operations::{initialised, uninitialised};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Error, Result};
use core::iter::repeat_with;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// This operation will perform a weighted sum of the parameters with the
/// input assuming they're of compatible shapes.
#[derive(Debug, Eq, PartialEq)]
pub struct Operation {
    pub(crate) output_neurons: u16,
}

impl Operation {
    /// This function constructs a new weighted sum operation with the
    /// given number of neurons to output from the operation.
    #[must_use]
    pub const fn new(output_neurons: u16) -> Self {
        Self { output_neurons }
    }
}

impl Sealed for Operation {}
impl uninitialised::Operation for Operation {
    type Initialised = initialised::weight_multiply::Operation;

    fn with_iter_private(
        self,
        iter: &mut impl Iterator<Item = ElementType>,
        input_neuron_count: u16,
    ) -> Result<(Self::Initialised, u16)> {
        let weight_dim = (input_neuron_count as usize, self.output_neurons as usize);
        let weight_count = weight_dim.0 * weight_dim.1;
        let parameter =
            Tensor::<rank::Two>::new(weight_dim, iter.take(weight_count)).map_err(|_| Error(()))?;
        let output_neurons = self.output_neurons;
        Ok((
            initialised::weight_multiply::Operation {
                input_neurons: input_neuron_count,
                parameter,
            },
            output_neurons,
        ))
    }

    fn with_seed_private(self, seed: u64, input_neuron_count: u16) -> (Self::Initialised, u16) {
        let mut generator = StdRng::seed_from_u64(seed);
        let xavier_delta = ElementType::sqrt(6.0)
            / ElementType::sqrt((input_neuron_count + self.output_neurons).into());
        // see Xavier initialization
        let mut iter = repeat_with(|| generator.gen_range(-xavier_delta..=xavier_delta));
        self.with_iter_private(&mut iter, input_neuron_count)
            .unwrap() // unwrapping is safe because we're generating an infinite sequence so there's always enough
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::UninitialisedOperation;

    #[test]
    fn test_new() {
        // Arrange
        let expected = Operation { output_neurons: 42 };

        // Act
        let operation = Operation::new(42);

        // Assert
        assert_eq!(operation, expected);
    }

    #[test]
    fn test_with_iter_private_success() {
        // Arrange
        let mut iter = [7.0, 8.0, 9.0].into_iter();
        let parameter = Tensor::<rank::Two>::new((3, 1), [7.0, 8.0, 9.0]).unwrap();
        let operation = Operation::new(1);
        let expected = initialised::weight_multiply::Operation {
            parameter,
            input_neurons: 3,
        };

        // Act
        let (operation, output_neurons) = operation.with_iter_private(&mut iter, 3).unwrap();

        // Assert
        assert_eq!(output_neurons, 1);
        assert_eq!(operation, expected);
    }

    #[test]
    fn test_with_iter_private_failure() {
        // Arrange
        let mut iter = [7.0, 8.0].into_iter();
        let operation = Operation::new(1);

        // Act
        let result = operation.with_iter_private(&mut iter, 3);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_with_seed_private() {
        // Arrange
        let operation = Operation::new(1);
        #[cfg(not(feature = "f32"))]
        let parameter = Tensor::<rank::Two>::new(
            (3, 1),
            [0.06505210094719227, 0.10465496341600944, 0.3342698606008603],
        )
        .unwrap();
        #[cfg(feature = "f32")]
        let parameter =
            Tensor::<rank::Two>::new((3, 1), [-0.8979591, 0.06505203, -0.61546296]).unwrap();
        let expected = initialised::weight_multiply::Operation {
            parameter,
            input_neurons: 3,
        };

        // Act
        let (operation, output_neurons) = operation.with_seed_private(42, 3);

        // Assert
        assert_eq!(output_neurons, 1);
        assert_eq!(operation, expected);
    }
}
