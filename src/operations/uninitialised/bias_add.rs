use crate::operations::{initialised, UninitialisedOperation};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Result};
use core::iter::repeat_with;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// This operation performs the bias addition portion of a dense layer.
/// The bias is a tensor that is added in at the end of the weighted sum
/// before passing through an activation function.
#[derive(Debug, Eq, PartialEq)]
pub struct Operation {
    xavier_lower_neuron_count: u16, // used as the "previous" neuron count for xavier initialisation
}

impl Operation {
    /// Constructs a new bias addition operation. We need to specify
    /// the number of neurons for the xavier generation to be correctly used
    /// to initialise the weights.
    #[must_use]
    pub const fn new(xavier_lower_neuron_count: u16) -> Self {
        Self {
            xavier_lower_neuron_count,
        }
    }
}

impl Sealed for Operation {}
impl UninitialisedOperation for Operation {
    type Initialised = initialised::bias_add::Operation;

    fn with_iter_private(
        self,
        iter: &mut impl Iterator<Item = ElementType>,
        input_neuron_count: u16,
    ) -> Result<(Self::Initialised, u16)> {
        let weight_dim = (1, input_neuron_count as usize);
        let weight_count = weight_dim.0 * weight_dim.1;
        let parameter = Tensor::<rank::Two>::new(weight_dim, iter.take(weight_count))?;
        Ok((
            initialised::bias_add::Operation { parameter },
            input_neuron_count,
        ))
    }

    fn with_seed_private(self, seed: u64, input_neuron_count: u16) -> (Self::Initialised, u16) {
        let mut generator = StdRng::seed_from_u64(seed);
        let xavier_delta = ElementType::sqrt(6.0)
            / ElementType::sqrt((self.xavier_lower_neuron_count + input_neuron_count).into());
        // see Xavier initialization
        let mut iter = repeat_with(|| generator.gen_range(-xavier_delta..=xavier_delta));
        self.with_iter_private(&mut iter, input_neuron_count)
            .unwrap() // unwrapping is safe because we're generating an infinite sequence so there's always enough
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_iter_private_success() {
        // Arrange
        let operation = Operation::new(3);
        let mut iter = [1.0, 2.0, 3.0, 4.0, 5.0].into_iter();
        let input_neuron_count = 5;

        // Act
        let (operation, neuron_count) = operation
            .with_iter_private(&mut iter, input_neuron_count)
            .unwrap();

        // Assert
        assert_eq!(operation.parameter.0.ncols(), neuron_count as usize);
        assert_eq!(input_neuron_count, neuron_count);
    }

    #[test]
    fn test_with_iter_private_failure() {
        // Arrange
        let operation = Operation::new(3);
        let mut iter = [1.0, 2.0, 3.0].into_iter();
        let input_neuron_count = 5;

        // Act
        let result = operation.with_iter_private(&mut iter, input_neuron_count);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_with_seed() {
        // Arrange
        let operation = Operation::new(3);
        let seed = 42;
        let input_neuron_count = 5;

        #[cfg(feature = "f32")]
        let expected = Tensor::<rank::Two>::new(
            (1, 5),
            [-0.63495296, 0.045998752, -0.435198, 0.074002385, 0.63813376],
        )
        .unwrap();
        #[cfg(not(feature = "f32"))]
        let expected = Tensor::<rank::Two>::new(
            (1, 5),
            [
                0.04599878171019156,
                0.07400223431629038,
                0.23636448517715036,
                -0.16298293564719446,
                -0.8065418982113659,
            ],
        )
        .unwrap();

        // Act
        let (operation, neuron_count) = operation.with_seed_private(seed, input_neuron_count);

        // Assert
        assert_eq!(operation.parameter, expected);
        assert_eq!(neuron_count, input_neuron_count);
    }
}
