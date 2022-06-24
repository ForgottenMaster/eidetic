//! This module contains the generic type representing a tensor of a certain rank.
//!
//! In deep learning, a tensor is simply an n-dimensional array. Different operations expect differing
//! dimensionality of tensor, so we make sure the dimensionality of the tensor is included in the type.

mod rank;
mod try_construct_tensor;

pub use rank::*;
pub use try_construct_tensor::*;

use ndarray::Array;

/// Represents a tensor with a specific element type T, and specific dimensionality
/// given by the R generic type parameter.
///
/// A tensor in machine learning is just a multi-dimensional array, sometimes referred
/// to as an "nd-array" for "n-dimensional array".
///
/// All operations in Eidetic will use the rank of the tensor to check at compile time
/// if operations and layers are connected correctly and won't allow mismatching ranks to
/// be connected depending on what the layer input/output supports.
#[derive(Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_iter() {
        // Arrange
        let tensor = (2, 3).try_construct_tensor(1..=6).unwrap();

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
        let mut tensor = (2, 3).try_construct_tensor(1..=6).unwrap();

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
        let tensor = (2, 3).try_construct_tensor(1..=6).unwrap();

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
}
