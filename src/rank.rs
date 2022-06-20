use ndarray::{Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5};

/// This trait represents the rank of a Tensor in Eidetic
/// which has a specific shape to define it. The rank of the tensor
/// is the number of dimensions that it has and can be found in the
/// type of the generic Tensor struct as a type parameter.
///
/// Note that this trait is sealed as the supertrait is not in the
/// public API meaning that all implementations for Rank exist solely
/// inside the Eidetic library.
pub trait Rank: RankPrivate {}

pub trait RankPrivate {
    type Internal: Dimension;
}

/// This is a unit struct that can be used to identify a rank 0 tensor.
#[derive(Debug)]
pub struct Rank0;
impl RankPrivate for Rank0 {
    type Internal = Ix0;
}
impl Rank for Rank0 {}

/// This is a unit struct that can be used to identify a rank 1 tensor.
#[derive(Debug)]
pub struct Rank1;
impl RankPrivate for Rank1 {
    type Internal = Ix1;
}
impl Rank for Rank1 {}

/// This is a unit struct that can be used to identify a rank 2 tensor.
#[derive(Debug)]
pub struct Rank2;
impl RankPrivate for Rank2 {
    type Internal = Ix2;
}
impl Rank for Rank2 {}

/// This is a unit struct that can be used to identify a rank 3 tensor.
#[derive(Debug)]
pub struct Rank3;
impl RankPrivate for Rank3 {
    type Internal = Ix3;
}
impl Rank for Rank3 {}

/// This is a unit struct that can be used to identify a rank 4 tensor.
#[derive(Debug)]
pub struct Rank4;
impl RankPrivate for Rank4 {
    type Internal = Ix4;
}
impl Rank for Rank4 {}

/// This is a unit struct that can be used to identify a rank 5 tensor.
#[derive(Debug)]
pub struct Rank5;
impl RankPrivate for Rank5 {
    type Internal = Ix5;
}
impl Rank for Rank5 {}
