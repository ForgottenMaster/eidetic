use crate::{Result, Tensor};

/// This trait defines the operations that must be supported by a given concrete backend
/// such as ndarray or arrayfire in order to be usable by the Eidetic framework.
///
/// The Eidetic framework operations and layers will operate with the abstract Backend
/// and Tensor traits but importantly callers should be able to rely on the results being the same
/// between backends.
///
/// See [`Tensor`] for details about the data layout.
///
/// For different backends that support the same data storage type, the user should be free to
/// select a different backend without changing any of the Eidetic framework, and the results should
/// be identical.
///
/// # Generics
/// T is the underlying data type for elements in the tensors we're operating on.
pub trait Backend<T> {
    /// The concrete tensor type in use by the backend for elements of
    /// type T. All operations will be performed using operands of this tensor type.
    type Tensor: Tensor<T>;

    /// Tries to create a new tensor from a given 4D shape tuple and an iterator
    /// of elements.
    ///
    /// # Errors
    /// Backend should return an error of type eidetic::Error::TensorConstruction if there are insufficient elements in the iterator.
    fn create_tensor(
        &self,
        shape: (usize, usize, usize, usize),
        iter: impl Iterator<Item = T>,
    ) -> Result<Self::Tensor>;
}
