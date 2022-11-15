use crate::Result;

/// This trait should be implemented by whichever type Eidetic should use
/// to represent the tensors it is operating on. For simplicity Eidetic assumes
/// all tensors to be rank 4 with the following structure (batch, layer, row, column/feature).
///
/// Therefore in order to access the value of the green channel (2nd channel) of the 9th pixel of the 4th row
/// of the 5th image in the batch we can use an index such as:
///
/// (4, 1, 3, 8)
///
/// Where:
/// - 4 => The index of the image in the batch
/// - 1 => The index of the color channel we're accessing
/// - 3 => The index of the row within the image
/// - 8 => The index of the feature/column within the row
///
/// When converting external data into and out of Tensor form for calling code
/// they can rely on this convention, and backends/implementations should arrange data
/// in such a way as to be correct when it's retrieved (e.g. internally converting this row
/// major form to column major).
///
/// # Generics
/// T is the underlying data type stored in the tensor.
pub trait Tensor<T>: IntoIterator<Item = T> + Sized {
    /// This function constructs a new instance of the Tensor from an iterator over its elements
    /// and a given set of dimensions for the constructed Tensor.
    ///
    /// # Errors
    /// Returns an error if the product of the dimensions provided is greater than the amount of elements
    /// that the iterator yields.
    fn from_iter(dims: (usize, usize, usize, usize), iter: impl Iterator<Item = T>)
        -> Result<Self>;
}
