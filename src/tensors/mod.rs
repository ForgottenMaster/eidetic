//! This module contains the generic type representing a tensor of a certain rank.
//!
//! In deep learning, a tensor is simply an n-dimensional array. Different operations expect differing
//! dimensionality of tensor, so we make sure the dimensionality of the tensor is included in the type.

pub mod rank;

use crate::{Error, Result};
use ndarray::{arr0, Array};
use rank::Rank;

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
    /// `Error` if the provided number of elements does not match the requested shape.
    pub fn new(shape: (usize, usize), iter: impl IntoIterator<Item = T>) -> Result<Self> {
        Array::from_iter(iter)
            .into_shape(shape)
            .map_err(|_| Error(()))
            .map(|array| Self(array))
    }
}

impl<T> Tensor<T, rank::Three> {
    /// Attempts to construct a rank 3 tensor of the specified shape from
    /// any iterable.
    ///
    /// # Errors
    /// `Error` if the provided number of elements does not match the requested shape.
    pub fn new(shape: (usize, usize, usize), iter: impl IntoIterator<Item = T>) -> Result<Self> {
        Array::from_iter(iter)
            .into_shape(shape)
            .map_err(|_| Error(()))
            .map(|array| Self(array))
    }
}

impl<T> Tensor<T, rank::Four> {
    /// Attempts to construct a rank 4 tensor of the specified shape from
    /// any iterable.
    ///
    /// # Errors
    /// `Error` if the provided number of elements does not match the requested shape.
    pub fn new(
        shape: (usize, usize, usize, usize),
        iter: impl IntoIterator<Item = T>,
    ) -> Result<Self> {
        Array::from_iter(iter)
            .into_shape(shape)
            .map_err(|_| Error(()))
            .map(|array| Self(array))
    }
}

impl<T> Tensor<T, rank::Five> {
    /// Attempts to construct a rank 5 tensor of the specified shape from
    /// any iterable.
    ///
    /// # Errors
    /// `Error` if the provided number of elements does not match the requested shape.
    pub fn new(
        shape: (usize, usize, usize, usize, usize),
        iter: impl IntoIterator<Item = T>,
    ) -> Result<Self> {
        Array::from_iter(iter)
            .into_shape(shape)
            .map_err(|_| Error(()))
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
