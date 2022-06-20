use crate::{Rank, Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Tensor, TensorConstructionError};
use ndarray::Array;

/// This trait represents a type that can be used to construct a tensor of a particular
/// rank and shape, given an iterator of elements.
///
/// The rank will be baked into the implementation but it will be able to create an
/// appropriately shaped tensor from any iterator type.
pub trait TryConstructTensor: TryConstructTensorPrivate {
    /// The rank of the tensor that is constructed.
    type Rank: Rank;

    /// A function which takes an iterator of elements of a type T
    /// and produces either a correctly shaped Tensor instance, or a
    /// TensorConstructionError with details on why it couldn't be constructed.
    ///
    /// # Examples
    ///
    /// ```
    /// use eidetic::*;
    /// assert!(().try_construct_tensor([1]).is_ok());
    /// assert!((5,).try_construct_tensor(1..=5).is_ok());
    /// assert!((2, 4).try_construct_tensor(1..=8).is_ok());
    /// assert!((1, 2, 3).try_construct_tensor(1..=6).is_ok());
    /// assert!((2, 3, 4, 5).try_construct_tensor(1..=120).is_ok());
    /// assert!((2, 3, 4, 5, 6).try_construct_tensor(1..=720).is_ok());
    /// ```
    ///
    /// ```
    /// use eidetic::*;
    /// assert_eq!(().try_construct_tensor([1, 2]).unwrap_err(), TensorConstructionError::InvalidShape { expected: 1 });
    /// assert_eq!((5,).try_construct_tensor([1..=6]).unwrap_err(), TensorConstructionError::InvalidShape { expected: 5 });
    /// assert_eq!((2, 4).try_construct_tensor(1..=9).unwrap_err(), TensorConstructionError::InvalidShape { expected: 8 });
    /// assert_eq!((1, 2, 3).try_construct_tensor(1..=5).unwrap_err(), TensorConstructionError::InvalidShape { expected: 6 });
    /// assert_eq!((2, 3, 4, 5).try_construct_tensor(1..=10).unwrap_err(), TensorConstructionError::InvalidShape { expected: 120 });
    /// assert_eq!((2, 3, 4, 5, 6).try_construct_tensor(1..=42).unwrap_err(), TensorConstructionError::InvalidShape { expected: 720 });
    /// ```
    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError>;
}

pub trait TryConstructTensorPrivate {}

impl TryConstructTensorPrivate for () {}
impl TryConstructTensor for () {
    type Rank = Rank0;

    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError> {
        Array::from_iter(input)
            .into_shape((*self).clone())
            .map_err(|_| TensorConstructionError::InvalidShape { expected: 1 })
            .map(|array| Tensor(array))
    }
}

impl TryConstructTensorPrivate for (usize,) {}
impl TryConstructTensor for (usize,) {
    type Rank = Rank1;

    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError> {
        Array::from_iter(input)
            .into_shape((*self).clone())
            .map_err(|_| TensorConstructionError::InvalidShape { expected: self.0 })
            .map(|array| Tensor(array))
    }
}

impl TryConstructTensorPrivate for (usize, usize) {}
impl TryConstructTensor for (usize, usize) {
    type Rank = Rank2;

    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError> {
        Array::from_iter(input)
            .into_shape((*self).clone())
            .map_err(|_| TensorConstructionError::InvalidShape {
                expected: self.0 * self.1,
            })
            .map(|array| Tensor(array))
    }
}

impl TryConstructTensorPrivate for (usize, usize, usize) {}
impl TryConstructTensor for (usize, usize, usize) {
    type Rank = Rank3;

    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError> {
        Array::from_iter(input)
            .into_shape((*self).clone())
            .map_err(|_| TensorConstructionError::InvalidShape {
                expected: self.0 * self.1 * self.2,
            })
            .map(|array| Tensor(array))
    }
}

impl TryConstructTensorPrivate for (usize, usize, usize, usize) {}
impl TryConstructTensor for (usize, usize, usize, usize) {
    type Rank = Rank4;

    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError> {
        Array::from_iter(input)
            .into_shape((*self).clone())
            .map_err(|_| TensorConstructionError::InvalidShape {
                expected: self.0 * self.1 * self.2 * self.3,
            })
            .map(|array| Tensor(array))
    }
}

impl TryConstructTensorPrivate for (usize, usize, usize, usize, usize) {}
impl TryConstructTensor for (usize, usize, usize, usize, usize) {
    type Rank = Rank5;

    fn try_construct_tensor<T>(
        &self,
        input: impl IntoIterator<Item = T>,
    ) -> Result<Tensor<T, Self::Rank>, TensorConstructionError> {
        Array::from_iter(input)
            .into_shape((*self).clone())
            .map_err(|_| TensorConstructionError::InvalidShape {
                expected: self.0 * self.1 * self.2 * self.3 * self.4,
            })
            .map(|array| Tensor(array))
    }
}
