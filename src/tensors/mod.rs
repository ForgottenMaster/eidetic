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
#[derive(Clone, Debug, PartialEq)]
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
