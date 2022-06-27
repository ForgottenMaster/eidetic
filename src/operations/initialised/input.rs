use crate::operations::InitialisedOperation;
use crate::operations::TrainableOperation;
use crate::operations::{initialised, trainable};
use crate::optimisers::base::OptimiserFactory;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Error, Result};
use core::iter::{empty, Empty};
use core::marker::PhantomData;

#[derive(Debug, Eq, PartialEq)]
pub struct Operation<T> {
    pub(crate) neurons: usize,
    pub(crate) phantom_data: PhantomData<T>,
}

impl<T> Sealed for Operation<T> {}
impl<T: OptimiserFactory<Tensor<rank::Two>>> InitialisedOperation<T> for Operation<T> {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type Parameter = Tensor<rank::Two>;
    type ParameterIter = Empty<ElementType>;
    type Trainable = trainable::input::Operation<T>;

    fn iter(&self) -> Self::ParameterIter {
        empty()
    }

    fn predict(&self, input: Self::Input) -> Result<Self::Output> {
        if input.0.ncols() == self.neurons {
            Ok(input)
        } else {
            Err(Error(()))
        }
    }

    fn with_optimiser<U>(self, _factory: U) -> <Self as initialised::Operation<U>>::Trainable
    where
        Self: initialised::Operation<U>,
        U: OptimiserFactory<<Self as initialised::Operation<U>>::Parameter>,
    {
        <Self as initialised::Operation<U>>::Trainable::new(self)
    }
}
