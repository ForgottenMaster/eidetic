use crate::private::Sealed;
use crate::tensors::{Five, Four, One, Rank, Tensor, Three, Two, Zero};
use ndarray::{arr0, Array};

#[cfg(feature = "thiserror")]
use thiserror::Error;

/// This trait represents a type that can be used to construct a tensor of a particular
/// rank and shape, given an iterator of elements.
///
/// The rank will be baked into the implementation but it will be able to create an
/// appropriately shaped tensor from any iterator type.
pub trait TryConstructTensor: Sealed {
    /// The rank of the tensor that is constructed.
    type Rank: Rank;

    /// A function which takes an iterator of elements of a type T
    /// and tries to construct a Tensor of the appropriate rank.
    ///
    /// # Errors
    /// `TensorConstructionError` if there was some issue with the construction from the given iterator.
    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError>;
}

impl Sealed for () {}
impl TryConstructTensor for () {
    type Rank = Zero;

    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError> {
        let mut input = input.into_iter();
        input.next().map_or(
            Err(TensorConstructionError::InvalidShape { expected: 1 }),
            |elem| {
                if input.next().is_none() {
                    Ok(Tensor(arr0(elem)))
                } else {
                    Err(TensorConstructionError::InvalidShape { expected: 1 })
                }
            },
        )
    }
}

impl Sealed for (usize,) {}
impl TryConstructTensor for (usize,) {
    type Rank = One;

    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError> {
        Array::from_iter(input)
            .into_shape(*self)
            .map_err(|_| TensorConstructionError::InvalidShape { expected: self.0 })
            .map(|array| Tensor(array))
    }
}

impl Sealed for (usize, usize) {}
impl TryConstructTensor for (usize, usize) {
    type Rank = Two;

    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError> {
        Array::from_iter(input)
            .into_shape(*self)
            .map_err(|_| TensorConstructionError::InvalidShape {
                expected: self.0 * self.1,
            })
            .map(|array| Tensor(array))
    }
}

impl Sealed for (usize, usize, usize) {}
impl TryConstructTensor for (usize, usize, usize) {
    type Rank = Three;

    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError> {
        Array::from_iter(input)
            .into_shape(*self)
            .map_err(|_| TensorConstructionError::InvalidShape {
                expected: self.0 * self.1 * self.2,
            })
            .map(|array| Tensor(array))
    }
}

impl Sealed for (usize, usize, usize, usize) {}
impl TryConstructTensor for (usize, usize, usize, usize) {
    type Rank = Four;

    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError> {
        Array::from_iter(input)
            .into_shape(*self)
            .map_err(|_| TensorConstructionError::InvalidShape {
                expected: self.0 * self.1 * self.2 * self.3,
            })
            .map(|array| Tensor(array))
    }
}

impl Sealed for (usize, usize, usize, usize, usize) {}
impl TryConstructTensor for (usize, usize, usize, usize, usize) {
    type Rank = Five;

    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError> {
        Array::from_iter(input)
            .into_shape(*self)
            .map_err(|_| TensorConstructionError::InvalidShape {
                expected: self.0 * self.1 * self.2 * self.3 * self.4,
            })
            .map(|array| Tensor(array))
    }
}

/// This is the error type which is used to report a failure to construct a new
/// tensor from a provided iterator of elements.
#[non_exhaustive]
#[derive(Debug, Eq, PartialEq)]
#[cfg_attr(feature = "thiserror", derive(Error))]
pub enum TensorConstructionError {
    /// This variant is used in the case that the product of the specified tensor shape components (e.g. rows * columns for a rank 2 tensor) doesn't match the number
    /// of elements provided in the iterator.
    #[cfg_attr(feature="thiserror", error("The provided iterator does not have the correct number of elements. Expected {expected} elements."))]
    InvalidShape {
        /// The number of elements that we expected to get.
        expected: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construct_rank_0_tensor_with_success() {
        // Arrange
        let shape = ();
        let input = [1];

        // Act
        shape.try_construct_tensor(input).unwrap();
    }

    #[test]
    fn test_construct_rank_1_tensor_with_success() {
        // Arrange
        let shape = (4,);
        let input = 1..=4;

        // Act
        shape.try_construct_tensor(input).unwrap();
    }

    #[test]
    fn test_construct_rank_2_tensor_with_success() {
        // Arrange
        let shape = (4, 3);
        let input = 1..=12;

        // Act
        shape.try_construct_tensor(input).unwrap();
    }

    #[test]
    fn test_construct_rank_3_tensor_with_success() {
        // Arrange
        let shape = (4, 3, 2);
        let input = 1..=24;

        // Act
        shape.try_construct_tensor(input).unwrap();
    }

    #[test]
    fn test_construct_rank_4_tensor_with_success() {
        // Arrange
        let shape = (4, 3, 2, 2);
        let input = 1..=48;

        // Act
        shape.try_construct_tensor(input).unwrap();
    }

    #[test]
    fn test_construct_rank_5_tensor_with_success() {
        // Arrange
        let shape = (4, 3, 2, 2, 3);
        let input = 1..=144;

        // Act
        shape.try_construct_tensor(input).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_construct_rank_0_tensor_with_failure_too_many() {
        // Arrange
        let shape = ();
        let input = [1, 2];

        // Act
        shape.try_construct_tensor(input).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_construct_rank_0_tensor_with_failure_too_few() {
        // Arrange
        let shape = ();
        let input: [u32; 0] = [];

        // Act
        shape.try_construct_tensor(input).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_construct_rank_1_tensor_with_failure() {
        // Arrange
        let shape = (4,);
        let input = 1..=5;

        // Act
        shape.try_construct_tensor(input).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_construct_rank_2_tensor_with_failure() {
        // Arrange
        let shape = (4, 3);
        let input = 1..=14;

        // Act
        shape.try_construct_tensor(input).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_construct_rank_3_tensor_with_failure() {
        // Arrange
        let shape = (4, 3, 2);
        let input = 1..=26;

        // Act
        shape.try_construct_tensor(input).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_construct_rank_4_tensor_with_failure() {
        // Arrange
        let shape = (4, 3, 2, 2);
        let input = 1..=42;

        // Act
        shape.try_construct_tensor(input).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_construct_rank_5_tensor_with_failure() {
        // Arrange
        let shape = (4, 3, 2, 2, 3);
        let input = 1..=154;

        // Act
        shape.try_construct_tensor(input).unwrap();
    }
}
