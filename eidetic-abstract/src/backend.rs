use crate::tensor::*;

/// This trait defines the operations that must be supported by a given concrete backend
/// such as ndarray or arrayfire in order to be usable by the Eidetic framework.
pub trait Backend {
    /// The associated type to use when we want to represent a rank 0 tensor
    /// AKA a scalar value.
    ///
    /// # Generics
    /// T is the underlying data type stored in the tensor.
    type Tensor0<T>: Tensor0<T>;

    /// The associated type to use when we want to represent a rank 1 tensor
    /// AKA a vector.
    ///
    /// # Generics
    /// T is the underlying data type stored in the tensor.
    type Tensor1<T>: Tensor1<T>;

    /// The associated type to use when we want to represent a rank 2 tensor
    /// AKA a matrix.
    ///
    /// # Generics
    /// T is the underlying data type stored in the tensor.
    type Tensor2<T>: Tensor2<T>;

    /// The associated type to use when we want to represent a rank 3 tensor
    /// AKA a vector of matrices.
    ///
    /// # Generics
    /// T is the underlying data type stored in the tensor.
    type Tensor3<T>: Tensor3<T>;

    /// The associated type to use when we want to represent a rank 4 tensor
    /// AKA a matrix of matrices.
    ///
    /// # Generics
    /// T is the underlying data type stored in the tensor.
    type Tensor4<T>: Tensor4<T>;
}
