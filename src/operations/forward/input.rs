use crate::operations::{backward, forward, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

impl<'a> forward::Construct<'a> for trainable::input::Operation {
    type Forward = Forward<'a>;
    fn construct(&'a mut self) -> Self::Forward {
        Forward::<'a>(self)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Forward<'a>(&'a mut trainable::input::Operation);

impl Sealed for Forward<'_> {}
impl<'a> forward::Operation for Forward<'a> {
    type Output = Tensor<rank::Two>;
    type Input = Tensor<rank::Two>;
    type Backward = backward::input::Operation;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        let neurons = output_gradient.0.ncols();
        let expected_neurons = self.0 .0.neurons;
        if neurons == expected_neurons {
            Ok((backward::input::Operation(()), output_gradient))
        } else {
            Err(Error(()))
        }
    }
}
