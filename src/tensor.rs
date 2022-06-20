//! This module contains the generic type representing a tensor of a certain rank.
//!
//! In deep learning, a tensor is simply an n-dimensional array. Different operations expect differing
//! dimensionality of tensor, so we make sure the dimensionality of the tensor is included in the type.

use crate::Rank;
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
    /// provided in try_from_iter for construction (row-major).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    /// Obtains an iterator over **mutable** references to the elements of the Tensor in the same ordering as they were
    /// provided in try_from_iter for construction (row-major).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut()
    }
}

// IntoIterator implementation for iterating over an owned tensor.
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
