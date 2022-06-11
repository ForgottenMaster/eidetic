//! This module contains any of the primitive types that will be useful for defining the concept
//! of a "Tensor" for use with Eidetic networks and operations.
//!
//! In deep learning, a tensor is simply an n-dimensional array. Different operations expect differing
//! dimensionality of tensor, so we make sure the dimensionality of the tensor is included in the type.

use crate::Error;
use ndarray::{Array, Ix2};

/// Represents a 2 dimensional tensor with elements of type T.
/// A 2 dimensional tensor is commonly referred to as a matrix and contains
/// multiple rows and columns. This is the most common tensor type found in the
/// operations within Eidetic.
#[derive(Debug)]
pub struct Tensor2<T>(Array<T, Ix2>);

impl<T> Tensor2<T> {
    /// Takes input data along with a shape for a 2-dimensional Tensor, and tries to construct
    /// the tensor.
    /// This will produce an error if the specified shape does not match the number of input elements.
    ///
    /// # Panics
    /// 1. Panics if the length of the data is greater than isize::MAX
    ///
    /// # Examples
    /// ```
    /// use eidetic::Tensor2;
    /// let result = Tensor2::try_from_iter((3, 3), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// assert!(result.is_ok());
    /// ```
    ///
    /// ```
    /// use eidetic::{Error, Tensor2};
    /// let result = Tensor2::try_from_iter((3, 2), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// assert!(result.is_err());
    /// assert_eq!(result.unwrap_err(), Error::TensorTryFromIterError {expected: 6, actual: 9});
    /// ```
    pub fn try_from_iter(
        shape: (usize, usize),
        iter: impl IntoIterator<Item = T>,
    ) -> Result<Self, Error> {
        let expected = shape.0 * shape.1;
        let flat_array = Array::from_iter(iter);
        let actual = flat_array.len();
        if expected != actual {
            Err(Error::TensorTryFromIterError { expected, actual })
        } else {
            let reshaped_array = flat_array.into_shape(shape).unwrap(); // safe to unwrap as we checked the shape previously
            Ok(Self(reshaped_array))
        }
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
        let tensor = Tensor2::try_from_iter((3, 3), input).unwrap();

        // Assert
        assert_eq!(tensor.0, expected);
    }

    #[test]
    fn test_incorrect_tensor2() {
        // Arrange
        let input = [1, 2, 3, 4, 5, 6, 7, 8, 9];

        // Act
        let err = Tensor2::try_from_iter((3, 2), input).unwrap_err();

        // Assert
        assert_eq!(
            err,
            Error::TensorTryFromIterError {
                expected: 6,
                actual: 9
            }
        );
    }
}
