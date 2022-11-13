//! This module contains the generic type representing a tensor of a certain rank.
//!
//! In deep learning, a tensor is simply an n-dimensional array. Different operations expect differing
//! dimensionality of tensor, so we make sure the dimensionality of the tensor is included in the type.

pub mod rank;

use crate::{ElementType, Error, Result};
use ndarray::{arr0, Array, Ix1, Ix2};
use rank::Rank;

/// Represents a tensor with a specific rank
/// given by the R generic type parameter.
///
/// A tensor in machine learning is just a multi-dimensional array, sometimes referred
/// to as an "nd-array" for "n-dimensional array".
///
/// All operations in Eidetic will use the rank of the tensor to check at compile time
/// if operations and layers are connected correctly and won't allow mismatching ranks to
/// be connected depending on what the layer input/output supports.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tensor<R: Rank>(pub(crate) Array<ElementType, R::Internal>);

impl Tensor<rank::Zero> {
    /// This function can be used to construct a new rank 0 tensor
    /// from a single element.
    #[must_use]
    pub fn new(elem: ElementType) -> Self {
        Self(arr0(elem))
    }
}

impl Tensor<rank::One> {
    /// This function constructs a rank 1 tensor from any iterable
    /// of elements.
    pub fn new(iter: impl IntoIterator<Item = ElementType>) -> Self {
        Self(Array::from_iter(iter))
    }
}

impl Tensor<rank::Two> {
    /// Attempts to construct a rank 2 tensor of the specified shape from
    /// any iterable.
    ///
    /// # Errors
    /// `Error` if the provided number of elements does not match the requested shape.
    pub fn new(shape: (usize, usize), iter: impl IntoIterator<Item = ElementType>) -> Result<Self> {
        let array: Array<ElementType, Ix1> = Array::from_iter(iter);
        let array: Array<ElementType, Ix2> = array.into_shape(shape).map_err(|_| Error(()))?;
        Ok(Self(array))
    }
}

impl Tensor<rank::Three> {
    /// Attempts to construct a rank 3 tensor of the specified shape from
    /// any iterable.
    ///
    /// # Errors
    /// `Error` if the provided number of elements does not match the requested shape.
    pub fn new(
        shape: (usize, usize, usize),
        iter: impl IntoIterator<Item = ElementType>,
    ) -> Result<Self> {
        Array::from_iter(iter)
            .into_shape(shape)
            .map_err(|_| Error(()))
            .map(Self)
    }
}

impl Tensor<rank::Four> {
    /// Attempts to construct a rank 4 tensor of the specified shape from
    /// any iterable.
    ///
    /// # Errors
    /// `Error` if the provided number of elements does not match the requested shape.
    pub fn new(
        shape: (usize, usize, usize, usize),
        iter: impl IntoIterator<Item = ElementType>,
    ) -> Result<Self> {
        Array::from_iter(iter)
            .into_shape(shape)
            .map_err(|_| Error(()))
            .map(Self)
    }
}

impl Tensor<rank::Five> {
    /// Attempts to construct a rank 5 tensor of the specified shape from
    /// any iterable.
    ///
    /// # Errors
    /// `Error` if the provided number of elements does not match the requested shape.
    pub fn new(
        shape: (usize, usize, usize, usize, usize),
        iter: impl IntoIterator<Item = ElementType>,
    ) -> Result<Self> {
        Array::from_iter(iter)
            .into_shape(shape)
            .map_err(|_| Error(()))
            .map(Self)
    }
}

/// This struct is the type that is returned from calling `into_iter()`
/// on a Tensor. This type is an Iterator that iterates the underlying elements.
pub struct TensorIterator<R: Rank>(<Array<ElementType, R::Internal> as IntoIterator>::IntoIter);

impl<R: Rank> IntoIterator for Tensor<R> {
    type Item = ElementType;
    type IntoIter = TensorIterator<R>;

    fn into_iter(self) -> Self::IntoIter {
        TensorIterator(self.0.into_iter())
    }
}

impl<R: Rank> Iterator for TensorIterator<R> {
    type Item = ElementType;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_rank_0_construction() {
        // Arrange
        let tensor = Tensor::<rank::Zero>::new(42.0);

        // Act
        let first = tensor.into_iter().next().unwrap();

        // Assert
        assert_eq!(first, 42.0);
    }

    #[test]
    fn test_tensor_rank_1_construction() {
        // Arrange
        let tensor = Tensor::<rank::One>::new((1..=3u16).map(|elem| ElementType::from(elem)));

        // Act
        let mut iter = tensor.into_iter();

        // Assert
        assert_eq!(iter.next().unwrap(), 1.0);
        assert_eq!(iter.next().unwrap(), 2.0);
        assert_eq!(iter.next().unwrap(), 3.0);
    }

    #[test]
    fn test_tensor_rank_2_construction() {
        // Arrange
        let tensor =
            Tensor::<rank::Two>::new((2, 2), (1..=4u16).map(|elem| ElementType::from(elem)))
                .unwrap();

        // Act
        let mut iter = tensor.into_iter();

        // Assert
        assert_eq!(iter.next().unwrap(), 1.0);
        assert_eq!(iter.next().unwrap(), 2.0);
        assert_eq!(iter.next().unwrap(), 3.0);
        assert_eq!(iter.next().unwrap(), 4.0);
    }

    #[test]
    fn test_tensor_rank_3_construction() {
        // Arrange
        let tensor =
            Tensor::<rank::Three>::new((2, 3, 2), (1..=12u16).map(|elem| ElementType::from(elem)))
                .unwrap();
        let expected = (1..=12u16).map(|elem| ElementType::from(elem));

        // Act
        let output = tensor.into_iter();

        // Assert
        assert!(expected.eq(output));
    }

    #[test]
    fn test_tensor_rank_4_construction() {
        // Arrange
        let tensor = Tensor::<rank::Four>::new(
            (2, 3, 2, 2),
            (1..=24u16).map(|elem| ElementType::from(elem)),
        )
        .unwrap();
        let expected = (1..=24u16).map(|elem| ElementType::from(elem));

        // Act
        let output = tensor.into_iter();

        // Assert
        assert!(expected.eq(output));
    }

    #[test]
    fn test_tensor_rank_5_construction() {
        // Arrange
        let tensor = Tensor::<rank::Five>::new(
            (2, 3, 2, 2, 3),
            (1..=72u16).map(|elem| ElementType::from(elem)),
        )
        .unwrap();
        let expected = (1..=72u16).map(|elem| ElementType::from(elem));

        // Act
        let output = tensor.into_iter();

        // Assert
        assert!(expected.eq(output));
    }

    #[test]
    fn test_tensor_rank_2_construction_failure() {
        // Arrange
        let tensor =
            Tensor::<rank::Two>::new((2, 2), (1..=5u16).map(|elem| ElementType::from(elem)));

        // Assert
        assert!(tensor.is_err());
    }

    #[test]
    fn test_tensor_rank_3_construction_failure() {
        // Arrange
        let tensor =
            Tensor::<rank::Three>::new((2, 3, 2), (1..=13u16).map(|elem| ElementType::from(elem)));

        // Assert
        assert!(tensor.is_err());
    }

    #[test]
    fn test_tensor_rank_4_construction_failure() {
        // Arrange
        let tensor = Tensor::<rank::Four>::new(
            (2, 3, 2, 4),
            (1..=51u16).map(|elem| ElementType::from(elem)),
        );

        // Assert
        assert!(tensor.is_err());
    }

    #[test]
    fn test_tensor_rank_5_construction_failure() {
        // Arrange
        let tensor = Tensor::<rank::Five>::new(
            (2, 3, 2, 4, 2),
            (1..=97u16).map(|elem| ElementType::from(elem)),
        );

        // Assert
        assert!(tensor.is_err());
    }
}
