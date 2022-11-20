//! This crate contains the abstract functionality for Eidetic which is independent of any specific
//! concrete Backend. This defines the way that networks can be built and trained, but delegates the
//! actual tensor storage and low-level operations used by layers to the backend currently in use.
//!
//! ```
//! todo!("Sample of building a network");
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(warnings, missing_docs, clippy::all)]

mod backend;
mod error;
mod tensor;

pub use backend::*;
pub use error::*;
pub use tensor::*;

#[cfg(test)]
pub use test::*;

#[cfg(test)]
mod test {
    use super::*;

    pub fn test_backend<T: Backend<f32> + Backend<f64>>(backend: T) {
        test_backend_impl(&backend, [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter());
        test_backend_impl(&backend, [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter());
    }

    fn test_backend_impl<T: PartialEq>(
        backend: &impl Backend<T>,
        input_data: impl Iterator<Item = T> + Clone,
    ) {
        test_tensor_create_failure(backend, input_data.clone());
        test_tensor_create_success(backend, input_data.clone());
    }

    fn test_tensor_create_failure<T>(
        backend: &impl Backend<T>,
        input_data: impl Iterator<Item = T>,
    ) {
        // Arrange
        let shape = (1, 1, 3, 3);

        // Act
        let result = backend.create_tensor(shape, input_data);

        // Assert
        assert!(matches!(
            result,
            Err(Error::TensorConstruction {
                requested_shape: (1, 1, 3, 3),
                number_of_elements: 6
            })
        ));
    }

    fn test_tensor_create_success<T: PartialEq>(
        backend: &impl Backend<T>,
        input_data: impl Iterator<Item = T> + Clone,
    ) {
        // Arrange
        let shape = (1, 1, 2, 3);

        // Act
        let output_data = backend.create_tensor(shape, input_data.clone()).unwrap();

        // Assert
        assert!(input_data.eq(output_data));
    }
}
