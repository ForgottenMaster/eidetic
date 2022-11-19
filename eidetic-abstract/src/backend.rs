use crate::TensorCreate;

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
pub trait Backend<T>: TensorCreate<T> {}
