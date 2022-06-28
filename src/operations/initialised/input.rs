use crate::operations::trainable;
use crate::operations::{InitialisedOperation, WithOptimiser};
use crate::optimisers::base::OptimiserFactory;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Error, Result};
use core::iter::{empty, Empty};

#[derive(Debug, Eq, PartialEq)]
pub struct Operation {
    pub(crate) neurons: usize,
}

impl Sealed for Operation {}
impl InitialisedOperation for Operation {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type ParameterIter = Empty<ElementType>;

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
}

impl<T: OptimiserFactory> WithOptimiser<T> for Operation {
    type Trainable = trainable::input::Operation;

    fn with_optimiser(self, _optimiser: T) -> Self::Trainable {
        trainable::input::Operation(self)
    }
}
