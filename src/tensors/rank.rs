//! This sub module contains the various supported ranks
//! that can be used by tensors.

use crate::private::Sealed;
use ndarray::{Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5};

/// This trait represents the rank of a Tensor in Eidetic
/// which has a specific shape to define it. The rank of the tensor
/// is the number of dimensions that it has and can be found in the
/// type of the generic Tensor struct as a type parameter.
///
/// Note that this trait is sealed as the supertrait is not in the
/// public API meaning that all implementations for Rank exist solely
/// inside the Eidetic library.
pub trait Rank: Clone + Sealed {
    #[doc(hidden)]
    type Internal: Dimension;
}

/// This is a unit struct that can be used to identify a rank 0 tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Zero;
impl Rank for Zero {
    type Internal = Ix0;
}
impl Sealed for Zero {}

/// This is a unit struct that can be used to identify a rank 1 tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct One;
impl Rank for One {
    type Internal = Ix1;
}
impl Sealed for One {}

/// This is a unit struct that can be used to identify a rank 2 tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Two;
impl Rank for Two {
    type Internal = Ix2;
}
impl Sealed for Two {}

/// This is a unit struct that can be used to identify a rank 3 tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Three;
impl Rank for Three {
    type Internal = Ix3;
}
impl Sealed for Three {}

/// This is a unit struct that can be used to identify a rank 4 tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Four;
impl Rank for Four {
    type Internal = Ix4;
}
impl Sealed for Four {}

/// This is a unit struct that can be used to identify a rank 5 tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Five;
impl Rank for Five {
    type Internal = Ix5;
}
impl Sealed for Five {}
