//! This module contains any of the primitive types that will be useful for defining the concept
//! of a "Tensor" for use with Eidetic networks and operations.
//!
//! In deep learning, a tensor is simply an n-dimensional array. Different operations expect differing
//! dimensionality of tensor, so we make sure the dimensionality of the tensor is included in the type.

use ndarray::Array;

/// This is a type which is used as a primitive type of holding and passing data through the
/// Eidetic API. It is unspecified how the Tensor holds its data, and any operations within
/// Eidetic will use this type. This ensures that the Eidetic API is treated as a kind of black box
/// where the input data must be prepared for use with Eidetic (converting into a Tensor), and after
/// being used with Eidetic must be extracted.
pub struct Tensor<S, D>(Array<S, D>);
