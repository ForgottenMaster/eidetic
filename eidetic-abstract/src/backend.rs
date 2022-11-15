use crate::{BackendDataType, Tensor};

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
pub trait Backend {
    /// The associated type to use when we want to represent our tensors.
    /// Must implement Tensor for the data type we're storing.
    ///
    /// # Generics
    /// T is the underlying data type stored in the tensor. Must be supported by this backend.
    type Tensor<T: BackendDataType<Self>>: Tensor<T>;
}
