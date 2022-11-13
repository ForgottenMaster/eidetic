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
    neurons: u16,
}

impl Operation {
    /// Constructs a new bias addition operation. We need to specify
    /// the number of neurons for the xavier generation to be correctly used
    /// to initialise the weights.
    #[must_use]
    pub const fn new(neurons: u16) -> Self {
        Self { neurons }
    }
}

impl Sealed for Operation {}
impl UninitialisedOperation for Operation {
    type Initialised = initialised::bias_add::Operation;

    fn with_iter_private(
        self,
        iter: &mut impl Iterator<Item = ElementType>,
        _input_neuron_count: u16,
    ) -> Result<(Self::Initialised, u16)> {
        let weight_dim = (1, self.neurons as usize);
        let weight_count = weight_dim.0 * weight_dim.1;
        let parameter = Tensor::<rank::Two>::new(weight_dim, iter.take(weight_count))?;
        Ok((initialised::bias_add::Operation { parameter }, self.neurons))
    }

    fn with_seed_private(self, seed: u64, input_neuron_count: u16) -> (Self::Initialised, u16) {
        let mut generator = StdRng::seed_from_u64(seed);
        let xavier_delta =
            ElementType::sqrt(6.0) / ElementType::sqrt((input_neuron_count + self.neurons).into());
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
        let operation = Operation::new(5);
        let mut iter = [1.0, 2.0, 3.0, 4.0, 5.0].into_iter();
        let input_neuron_count = 3;

        // Act
        let (operation, neuron_count) = operation
            .with_iter_private(&mut iter, input_neuron_count)
            .unwrap();

        // Assert
        assert_eq!(operation.parameter.0.ncols(), neuron_count as usize);
        assert_eq!(neuron_count, 5);
    }

    #[test]
    fn test_with_iter_private_failure() {
        // Arrange
        let operation = Operation::new(5);
        let mut iter = [1.0, 2.0, 3.0].into_iter();
        let input_neuron_count = 3;

        // Act
        let result = operation.with_iter_private(&mut iter, input_neuron_count);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_with_seed() {
        // Arrange
        let operation = Operation::new(5);
        let seed = 42;
        let input_neuron_count = 5;

        #[cfg(feature = "f32")]
        let expected = Tensor::<rank::Two>::new(
            (1, 5),
            [-0.5679192, 0.041142583, -0.38925293, 0.06618971, 0.5707642],
        )
        .unwrap();
        #[cfg(not(feature = "f32"))]
        let expected = Tensor::<rank::Two>::new(
            (1, 5),
            [
                0.041142561114465015,
                0.06618961056723727,
                0.2114108225291399,
                -0.14577636931184024,
                -0.7213930044409321,
            ],
        )
        .unwrap();

        // Act
        let (operation, neuron_count) = operation.with_seed_private(seed, input_neuron_count);

        // Assert
        assert_eq!(operation.parameter, expected);
        assert_eq!(neuron_count, 5);
    }
}
