//! This module contains the generic type representing a tensor of a certain rank.
//!
//! In deep learning, a tensor is simply an n-dimensional array. Different operations expect differing
//! dimensionality of tensor, so we make sure the dimensionality of the tensor is included in the type.

pub mod rank;

use ndarray::{arr0, Array};
use rank::Rank;
#[cfg(feature = "thiserror")]
use thiserror::Error;

/// Represents a tensor with a specific element type T, and specific dimensionality
/// given by the R generic type parameter.
///
/// A tensor in machine learning is just a multi-dimensional array, sometimes referred
/// to as an "nd-array" for "n-dimensional array".
///
/// All operations in Eidetic will use the rank of the tensor to check at compile time
/// if operations and layers are connected correctly and won't allow mismatching ranks to
/// be connected depending on what the layer input/output supports.
#[derive(Debug, PartialEq)]
pub struct Tensor<T, R: Rank>(pub(crate) Array<T, R::Internal>);

impl<T, R: Rank> Tensor<T, R> {
    /// Obtains an iterator over references to the elements of the Tensor2 in the same ordering as they were
    /// provided in `try_from_iter` for construction (row-major).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    /// Obtains an iterator over **mutable** references to the elements of the Tensor in the same ordering as they were
    /// provided in `try_from_iter` for construction (row-major).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut()
    }
}

impl<T> Tensor<T, rank::Zero> {
    /// This function can be used to construct a new rank 0 tensor
    /// from a single element.
    pub fn new(elem: T) -> Self {
        Self(arr0(elem))
    }
}

impl<T> Tensor<T, rank::One> {
    /// This function constructs a rank 1 tensor from any iterable
    /// of elements.
    pub fn new(iter: impl IntoIterator<Item = T>) -> Self {
        Self(Array::from_iter(iter))
    }
}

impl<T> Tensor<T, rank::Two> {
    /// Attempts to construct a rank 2 tensor of the specified shape from
    /// any iterable.
    ///
    /// # Errors
    /// `TensorConstructionError` if the construction fails such as, insufficient elements provided for the shape.
    pub fn new(
        shape: (usize, usize),
        iter: impl IntoIterator<Item = T>,
    ) -> Result<Self, TensorConstructionError> {
        Array::from_iter(iter)
            .into_shape(shape)
            .map_err(|_| TensorConstructionError::InvalidShape {
                expected: shape.0 * shape.1,
            })
            .map(|array| Self(array))
    }
}

impl<T> Tensor<T, rank::Three> {
    /// Attempts to construct a rank 3 tensor of the specified shape from
    /// any iterable.
    ///
    /// # Errors
    /// `TensorConstructionError` if the construction fails such as, insufficient elements provided for the shape.
    pub fn new(
        shape: (usize, usize, usize),
        iter: impl IntoIterator<Item = T>,
    ) -> Result<Self, TensorConstructionError> {
        Array::from_iter(iter)
            .into_shape(shape)
            .map_err(|_| TensorConstructionError::InvalidShape {
                expected: shape.0 * shape.1 * shape.2,
            })
            .map(|array| Self(array))
    }
}

impl<T> Tensor<T, rank::Four> {
    /// Attempts to construct a rank 4 tensor of the specified shape from
    /// any iterable.
    ///
    /// # Errors
    /// `TensorConstructionError` if the construction fails such as, insufficient elements provided for the shape.
    pub fn new(
        shape: (usize, usize, usize, usize),
        iter: impl IntoIterator<Item = T>,
    ) -> Result<Self, TensorConstructionError> {
        Array::from_iter(iter)
            .into_shape(shape)
            .map_err(|_| TensorConstructionError::InvalidShape {
                expected: shape.0 * shape.1 * shape.2 * shape.3,
            })
            .map(|array| Self(array))
    }
}

impl<T> Tensor<T, rank::Five> {
    /// Attempts to construct a rank 5 tensor of the specified shape from
    /// any iterable.
    ///
    /// # Errors
    /// `TensorConstructionError` if the construction fails such as, insufficient elements provided for the shape.
    pub fn new(
        shape: (usize, usize, usize, usize, usize),
        iter: impl IntoIterator<Item = T>,
    ) -> Result<Self, TensorConstructionError> {
        Array::from_iter(iter)
            .into_shape(shape)
            .map_err(|_| TensorConstructionError::InvalidShape {
                expected: shape.0 * shape.1 * shape.2 * shape.3 * shape.4,
            })
            .map(|array| Self(array))
    }
}

/// This struct is the type that is returned from calling `into_iter()`
/// on a Tensor. This type is an Iterator that iterates the underlying elements.
pub struct TensorIterator<T, R: Rank>(<Array<T, R::Internal> as IntoIterator>::IntoIter);

impl<T, R: Rank> IntoIterator for Tensor<T, R> {
    type Item = T;
    type IntoIter = TensorIterator<T, R>;

    fn into_iter(self) -> Self::IntoIter {
        TensorIterator(self.0.into_iter())
    }
}

impl<T, R: Rank> Iterator for TensorIterator<T, R> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
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
    fn test_tensor_iter() {
        // Arrange
        let tensor = Tensor::<_, rank::Two>::new((2, 3), 1..=6).unwrap();

        // Act
        let mut iter = tensor.iter();

        // Assert
        assert_eq!(iter.next().unwrap(), &1);
        assert_eq!(iter.next().unwrap(), &2);
        assert_eq!(iter.next().unwrap(), &3);
        assert_eq!(iter.next().unwrap(), &4);
        assert_eq!(iter.next().unwrap(), &5);
        assert_eq!(iter.next().unwrap(), &6);
    }

    #[test]
    fn test_tensor_iter_mut() {
        // Arrange
        let mut tensor = Tensor::<_, rank::Two>::new((2, 3), 1..=6).unwrap();

        // Act
        let mut iter = tensor.iter_mut();

        // Assert
        assert_eq!(iter.next().unwrap(), &mut 1);
        assert_eq!(iter.next().unwrap(), &mut 2);
        assert_eq!(iter.next().unwrap(), &mut 3);
        assert_eq!(iter.next().unwrap(), &mut 4);
        assert_eq!(iter.next().unwrap(), &mut 5);
        assert_eq!(iter.next().unwrap(), &mut 6);
    }

    #[test]
    fn test_tensor_into_iter() {
        // Arrange
        let tensor = Tensor::<_, rank::Two>::new((2, 3), 1..=6).unwrap();

        // Act
        let mut iter = tensor.into_iter();

        // Assert
        assert_eq!(iter.next().unwrap(), 1);
        assert_eq!(iter.next().unwrap(), 2);
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.next().unwrap(), 4);
        assert_eq!(iter.next().unwrap(), 5);
        assert_eq!(iter.next().unwrap(), 6);
    }

    #[test]
    fn test_construct_rank_0_tensor() {
        // Arrange
        let elem = 42;
        let expected = Tensor(arr0(elem));

        // Act
        let output = Tensor::<_, rank::Zero>::new(elem);

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_construct_rank_1_tensor() {
        // Arrange
        let iter = [1, 2, 3, 4, 5, 6];
        let expected = Tensor(Array::from_iter(iter.clone()));

        // Act
        let output = Tensor::<_, rank::One>::new(iter);

        // Assert
        assert_eq!(output, expected);
    }
}
