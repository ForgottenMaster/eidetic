use crate::operations::forward::Construct;
use crate::operations::InitialisedOperation;
use crate::operations::{forward, initialised, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::Result;

#[derive(Debug, PartialEq)]
pub struct Operation {
    pub(crate) initialised: initialised::sigmoid::Operation,
    pub(crate) last_output: Tensor<rank::Two>,
}

impl Sealed for Operation {}
impl trainable::Operation for Operation {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type Initialised = initialised::sigmoid::Operation;

    fn into_initialised(self) -> Self::Initialised {
        self.initialised
    }

    fn forward<'a>(
        &'a mut self,
        input: Self::Input,
    ) -> Result<(<Self as forward::Construct<'a>>::Forward, Self::Output)>
    where
        Self: forward::Construct<'a>,
    {
        self.last_output = self.initialised.predict(input)?;
        let clone = self.last_output.clone();
        Ok((self.construct(), clone))
    }
}
