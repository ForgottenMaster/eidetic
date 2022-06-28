use crate::operations::forward::Construct;
use crate::operations::{forward, initialised, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

#[derive(Debug, Eq, PartialEq)]
#[cfg_attr(tests, repr(C))] // code coverage hack, I dislike <100% in the report :(
pub struct Operation(pub(crate) initialised::input::Operation);

impl Sealed for Operation {}
impl trainable::Operation for Operation {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type Initialised = initialised::input::Operation;

    fn into_initialised(self) -> Self::Initialised {
        self.0
    }

    fn forward<'a>(
        &'a mut self,
        input: Self::Input,
    ) -> Result<(<Self as forward::Construct<'a>>::Forward, Self::Output)>
    where
        Self: forward::Construct<'a>,
    {
        if self.0.neurons == input.0.ncols() {
            Ok((self.construct(), input))
        } else {
            Err(Error(()))
        }
    }
}
