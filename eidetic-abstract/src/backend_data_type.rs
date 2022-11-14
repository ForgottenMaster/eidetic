use crate::Backend;

/// A trait that should be implemented for any types that a given backend
/// actually supports.
///
/// # Generics
/// T is the backend type for which the implementing type is a valid data type to be stored in tensors.
pub trait BackendDataType<T: Backend> {}
