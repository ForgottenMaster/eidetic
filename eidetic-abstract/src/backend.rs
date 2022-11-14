use crate::{BackendDataType, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4};

/// This trait defines the operations that must be supported by a given concrete backend
/// such as ndarray or arrayfire in order to be usable by the Eidetic framework.
pub trait Backend {
    /// The associated type to use when we want to represent a rank 0 tensor
    /// AKA a scalar value.
    ///
    /// # Generics
    /// T is the underlying data type stored in the tensor. Must be supported by this backend.
    type Tensor0<T: BackendDataType<Self>>: Tensor0<T>;

    /// The associated type to use when we want to represent a rank 1 tensor
    /// AKA a vector.
    ///
    /// # Generics
    /// T is the underlying data type stored in the tensor. Must be supported by this backend.
    type Tensor1<T: BackendDataType<Self>>: Tensor1<T>;

    /// The associated type to use when we want to represent a rank 2 tensor
    /// AKA a matrix.
    ///
    /// # Generics
    /// T is the underlying data type stored in the tensor. Must be supported by this backend.
    type Tensor2<T: BackendDataType<Self>>: Tensor2<T>;

    /// The associated type to use when we want to represent a rank 3 tensor
    /// AKA a vector of matrices.
    ///
    /// # Generics
    /// T is the underlying data type stored in the tensor. Must be supported by this backend.
    type Tensor3<T: BackendDataType<Self>>: Tensor3<T>;

    /// The associated type to use when we want to represent a rank 4 tensor
    /// AKA a matrix of matrices.
    ///
    /// # Generics
    /// T is the underlying data type stored in the tensor. Must be supported by this backend.
    type Tensor4<T: BackendDataType<Self>>: Tensor4<T>;
}
