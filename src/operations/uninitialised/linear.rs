//use crate::operations::uninitialised;
use crate::private::Sealed;
use crate::tensors::rank::Rank;
use crate::tensors::Tensor;
use core::marker::PhantomData;

/// This operation represents a linear passthrough operation
/// which is used for example as an activation function for a
/// network layer. It can also be used as an initial input operation
/// in order to have something to chain another operation against to
/// ensure that input/output neurons are calculated.
pub struct Linear<T, R: Rank>(PhantomData<Tensor<T, R>>);

impl<T, R: Rank> Sealed for Linear<T, R> {}
/*impl<T, R: Rank> uninitialised::Operation for Linear<T, R> {
    type Element = T;
}*/
