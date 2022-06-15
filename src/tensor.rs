//! This module contains any of the primitive types that will be useful for defining the concept
//! of a "Tensor" for use with Eidetic networks and operations.
//!
//! In deep learning, a tensor is simply an n-dimensional array. Different operations expect differing
//! dimensionality of tensor, so we make sure the dimensionality of the tensor is included in the type.

use crate::TensorConstructionError;
use ndarray::{Array, Ix2};

/// Represents a 2 dimensional tensor with elements of type T.
/// A 2 dimensional tensor is commonly referred to as a matrix and contains
/// multiple rows and columns. This is the most common tensor type found in the
/// operations within Eidetic.
///
/// Since the number of neurons is a core architectural decision, it's baked in at the type level.
/// This means to construct a new Tensor2 we only need specify the number of rows and the number of neurons
/// will be provided at the type level.
///
/// This will also be a useful property when chaining network operations together since they should have matching
/// expected neuron counts to allow the tensors to flow through the network properly.
#[derive(Debug)]
pub struct Tensor2<T, const NEURONS: usize>(pub(crate) Array<T, Ix2>);

impl<T, const NEURONS: usize> Tensor2<T, NEURONS> {
    /// Takes input data along with the number of rows for a 2-dimensional Tensor, and tries to construct
    /// the tensor.
    /// The number of neurons in the tensor is equivalent to the number of columns. As such, the specified
    /// shape only gives the row count.
    /// This will produce an error if the specified shape does not match the number of input elements.
    ///
    /// # Panics
    /// 1. Panics if the length of the data is greater than isize::MAX
    ///
    /// # Examples
    /// ```
    /// use eidetic::Tensor2;
    /// let result = Tensor2::<_, 3>::try_from_iter(3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// assert!(result.is_ok());
    /// ```
    ///
    /// ```
    /// use eidetic::{TensorConstructionError, Tensor2};
    /// let result = Tensor2::<_, 2>::try_from_iter(3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// assert!(result.is_err());
    /// assert_eq!(result.unwrap_err(), TensorConstructionError::InvalidShape {expected: 6, actual: 9});
    /// ```
    pub fn try_from_iter(
        rows: usize,
        iter: impl IntoIterator<Item = T>,
    ) -> Result<Self, TensorConstructionError> {
        let expected = rows * NEURONS;
        let flat_array = Array::from_iter(iter);
        let actual = flat_array.len();
        if expected != actual {
            Err(TensorConstructionError::InvalidShape { expected, actual })
        } else {
            let reshaped_array = flat_array.into_shape((rows, NEURONS)).unwrap(); // safe to unwrap as we checked the shape previously
            Ok(Self(reshaped_array))
        }
    }

    /// Obtains an iterator over references to the elements of the Tensor2 in the same ordering as they were
    /// provided in try_from_iter for construction (row-major).
    ///
    /// # Examples
    /// ```
    /// use eidetic::Tensor2;
    /// let tensor = Tensor2::<_, 3>::try_from_iter(2, [1, 2, 3, 4, 5, 6]).unwrap();
    /// let mut iter = tensor.iter();
    /// assert_eq!(iter.next().unwrap(), &1);
    /// assert_eq!(iter.next().unwrap(), &2);
    /// assert_eq!(iter.next().unwrap(), &3);
    /// assert_eq!(iter.next().unwrap(), &4);
    /// assert_eq!(iter.next().unwrap(), &5);
    /// assert_eq!(iter.next().unwrap(), &6);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    /// Obtains an iterator over **mutable** references to the elements of the Tensor2 in the same ordering as they were
    /// provided in try_from_iter for construction (row-major).
    ///
    /// # Examples
    /// ```
    /// use eidetic::Tensor2;
    /// let mut tensor = Tensor2::<_, 3>::try_from_iter(2, [1, 2, 3, 4, 5, 6]).unwrap();
    /// let mut iter = tensor.iter_mut();
    /// assert_eq!(iter.next().unwrap(), &mut 1);
    /// assert_eq!(iter.next().unwrap(), &mut 2);
    /// assert_eq!(iter.next().unwrap(), &mut 3);
    /// assert_eq!(iter.next().unwrap(), &mut 4);
    /// assert_eq!(iter.next().unwrap(), &mut 5);
    /// assert_eq!(iter.next().unwrap(), &mut 6);
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut()
    }
}

pub struct Tensor2Iterator<T>(<Array<T, Ix2> as IntoIterator>::IntoIter);

impl<T, const NEURONS: usize> IntoIterator for Tensor2<T, NEURONS> {
    type Item = T;
    type IntoIter = Tensor2Iterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        Tensor2Iterator(self.0.into_iter())
    }
}

impl<T> Iterator for Tensor2Iterator<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correct_tensor2() {
        // Arrange
        let input = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let expected = Array::from_iter([1, 2, 3, 4, 5, 6, 7, 8, 9])
            .into_shape((3, 3))
            .unwrap();

        // Act
        let tensor = Tensor2::<_, 3>::try_from_iter(3, input).unwrap();

        // Assert
        assert_eq!(tensor.0, expected);
    }

    #[test]
    fn test_incorrect_tensor2() {
        // Arrange
        let input = [1, 2, 3, 4, 5, 6, 7, 8, 9];

        // Act
        let err = Tensor2::<_, 2>::try_from_iter(3, input).unwrap_err();

        // Assert
        assert_eq!(
            err,
            TensorConstructionError::InvalidShape {
                expected: 6,
                actual: 9
            }
        );
    }

    #[test]
    fn test_tensor_2_iter() {
        // Arrange
        let tensor = Tensor2::<_, 3>::try_from_iter(3, [1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();

        // Act
        let mut iter = tensor.iter();

        // Assert
        assert_eq!(iter.next().unwrap(), &1);
        assert_eq!(iter.next().unwrap(), &2);
        assert_eq!(iter.next().unwrap(), &3);
        assert_eq!(iter.next().unwrap(), &4);
        assert_eq!(iter.next().unwrap(), &5);
    }

    #[test]
    fn test_tensor_2_iter_mut() {
        // Arrange
        let mut tensor = Tensor2::<_, 3>::try_from_iter(3, [1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();

        // Act
        let mut iter = tensor.iter_mut();

        // Assert
        assert_eq!(iter.next().unwrap(), &mut 1);
        assert_eq!(iter.next().unwrap(), &mut 2);
        assert_eq!(iter.next().unwrap(), &mut 3);
        assert_eq!(iter.next().unwrap(), &mut 4);
        assert_eq!(iter.next().unwrap(), &mut 5);
    }

    #[test]
    fn test_tensor_2_into_iter() {
        // Arrange
        let tensor = Tensor2::<_, 3>::try_from_iter(3, [1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();

        // Act
        let mut iter = tensor.into_iter();

        // Assert
        assert_eq!(iter.next().unwrap(), 1);
        assert_eq!(iter.next().unwrap(), 2);
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.next().unwrap(), 4);
        assert_eq!(iter.next().unwrap(), 5);
    }
}
