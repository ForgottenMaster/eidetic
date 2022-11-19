use crate::{Result, Tensor};

/// This trait represents the ability to create a tensor with element type T
/// from an iterator of those elements, and a (4D) shape tuple.
pub trait TensorCreate<T> {
    /// The output type of the tensor that is created on success.
    /// Must implement the Tensor<T> trait so as to be able to use the
    /// methods provided by that trait in an agnostic way.
    type Output: Tensor<T>;

    /// The function that constructs a tensor from the given shape and element stream.
    /// Returns an error if the given shape doesn't match the number of elements in the
    /// iterator.
    fn tensor_create(
        shape: (usize, usize, usize, usize),
        iter: impl Iterator<Item = T>,
    ) -> Result<Self::Output>;
}
