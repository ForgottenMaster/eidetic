/// This trait should be implemented by a type that represents a
/// rank 0 tensor, which is also known as a scalar value.
///
/// # Generics
/// T is the underlying data type stored in the tensor.
pub trait Tensor0<T> {}

/// This trait should be implemented by a type that represents a
/// rank 1 tensor, which is also known as a vector.
///
/// # Generics
/// T is the underlying data type stored in the tensor.
pub trait Tensor1<T> {}

/// This trait should be implemented by a type that represents a
/// rank 2 tensor, which is also known as a matrix.
///
/// # Generics
/// T is the underlying data type stored in the tensor.
pub trait Tensor2<T> {}

/// This trait should be implemented by a type that represents a
/// rank 3 tensor, which can be thought of as a vector of matrices
/// or a 3-D array.
///
/// # Generics
/// T is the underlying data type stored in the tensor.
pub trait Tensor3<T> {}

/// This trait should be implemented by a type that represents a
/// rank 4 tensor, which can be thought of as a matrix of matrices
/// or a 4-D array.
///
/// # Generics
/// T is the underlying data type stored in the tensor.
pub trait Tensor4<T> {}
