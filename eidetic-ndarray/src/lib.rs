//! This crate implements an Eidetic backend using the ndarray crate for tensor storage.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(warnings, missing_docs, clippy::all)]

use core::marker::PhantomData;
use ndarray::{Array, Ix4};

/// The structure that will be used as the backend for ndarray backend.
pub struct Backend;

/// The structure that will be used to represent a Tensor with the ndarray
/// backend.
///
/// # Generics
/// 'a is the lifetime of the Backend borrow
/// T is the data type of the underlying elements
pub struct Tensor<'a, T>(PhantomData<&'a ()>, Array<T, Ix4>);

macro_rules! implement_backend {
    ($type:ty) => {
        impl eidetic_abstract::Backend<$type> for Backend {
            type Tensor<'a> = Tensor<'a, $type>;

            fn create_tensor(
                &self,
                shape: (usize, usize, usize, usize),
                iter: impl Iterator<Item = $type>,
            ) -> eidetic_abstract::Result<Self::Tensor<'_>> {
                let expected = shape.0 * shape.1 * shape.2 * shape.3;
                let flat = Array::from_iter(iter.take(expected));
                let count = flat.len();
                if count != expected {
                    Err(eidetic_abstract::Error::TensorConstruction {
                        number_of_elements: count,
                        requested_shape: shape,
                    })
                } else {
                    Ok(Tensor(PhantomData, flat.into_shape(shape).unwrap()))
                }
            }
        }

        impl<'a> eidetic_abstract::Tensor<'a, $type> for Tensor<'a, $type> {}

        impl<'a> IntoIterator for Tensor<'a, $type> {
            type Item = $type;
            type IntoIter = <Array<$type, Ix4> as IntoIterator>::IntoIter;

            fn into_iter(self) -> Self::IntoIter {
                self.1.into_iter()
            }
        }
    };
}
implement_backend!(f32);
implement_backend!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend() {
        eidetic_abstract::test_backend(Backend);
    }
}
