use crate::operations::{forward, initialised, Forward, TrainableOperation};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::Result;
use core::iter::repeat_with;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};

#[derive(Debug, PartialEq)]
pub struct Operation {
    pub(crate) initialised: initialised::dropout::Operation,
}

impl Sealed for Operation {}

impl TrainableOperation for Operation {
    type Initialised = initialised::dropout::Operation;

    fn into_initialised(self) -> Self::Initialised {
        self.initialised
    }
}

impl<'a> Forward<'a> for Operation {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type Forward = forward::dropout::Operation<'a>;

    fn forward(&'a mut self, input: Self::Input) -> Result<(Self::Forward, Self::Output)> {
        let mut random = match self.initialised.seed {
            Some(seed) => {
                self.initialised.seed = Some(seed + 1); // so we don't get same mask next time
                StdRng::seed_from_u64(seed)
            }
            None => StdRng::from_rng(thread_rng()).unwrap(),
        };
        let dims = input.0.raw_dim();
        let element_count = dims[0] * dims[1];
        let keep_probability = self.initialised.keep_probability;
        let iter = repeat_with(|| random.gen_range(0.0..=1.0))
            .map(|elem| if elem <= keep_probability { 1.0 } else { 0.0 })
            .take(element_count);
        let mask = Tensor::<rank::Two>::new((dims[0], dims[1]), iter).unwrap();
        let output = Tensor(input.0 * &mask.0);
        let forward = Self::Forward {
            _borrow: self,
            mask,
        };
        Ok((forward, output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_initialised() {
        // Arrange
        let trainable = Operation {
            initialised: initialised::dropout::Operation {
                keep_probability: 0.8,
                seed: None,
            },
        };
        let expected = initialised::dropout::Operation {
            keep_probability: 0.8,
            seed: None,
        };

        // Act
        let output = trainable.into_initialised();

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_forward() {
        // Arrange
        let mut trainable = Operation {
            initialised: initialised::dropout::Operation {
                keep_probability: 0.6,
                seed: Some(42),
            },
        };
        let mut expected_backing = Operation {
            initialised: initialised::dropout::Operation {
                keep_probability: 0.6,
                seed: Some(43),
            },
        };
        let input = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();
        #[cfg(not(feature = "f32"))]
        let mask = Tensor::<rank::Two>::new((1, 3), [1.0, 1.0, 0.0]).unwrap();
        #[cfg(feature = "f32")]
        let mask = Tensor::<rank::Two>::new((1, 3), [1.0, 1.0, 1.0]).unwrap();
        let expected_forward = forward::dropout::Operation {
            _borrow: &mut expected_backing,
            mask,
        };
        #[cfg(not(feature = "f32"))]
        let expected_output = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 0.0]).unwrap();
        #[cfg(feature = "f32")]
        let expected_output = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();

        // Act
        let (forward, output) = trainable.forward(input).unwrap();

        // Assert
        assert_eq!(forward, expected_forward);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_forward_without_seed() {
        // Arrange
        let mut trainable = Operation {
            initialised: initialised::dropout::Operation {
                keep_probability: 0.6,
                seed: None,
            },
        };
        let input = Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap();

        // Act
        trainable.forward(input).unwrap();
    }
}
