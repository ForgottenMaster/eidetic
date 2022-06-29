use crate::operations::{backward, forward, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

impl<'a> forward::Construct<'a> for trainable::sigmoid::Operation {
    type Forward = Forward<'a>;
    fn construct(&'a mut self) -> Self::Forward {
        Forward::<'a>(self)
    }
}

#[derive(Debug, PartialEq)]
pub struct Forward<'a>(&'a mut trainable::sigmoid::Operation);

impl Sealed for Forward<'_> {}
impl<'a> forward::Operation for Forward<'a> {
    type Output = Tensor<rank::Two>;
    type Input = Tensor<rank::Two>;
    type Backward = backward::sigmoid::Operation;

    fn backward(self, output_gradient: Self::Output) -> Result<(Self::Backward, Self::Input)> {
        let neurons = output_gradient.0.ncols();
        let expected_neurons = self.0.initialised.neurons;
        if neurons == expected_neurons {
            let partial = self.0.last_output.0.mapv(|elem| elem * (1.0 - elem));
            let input_gradient = Tensor(partial * output_gradient.0);
            Ok((backward::sigmoid::Operation(()), input_gradient))
        } else {
            Err(Error(()))
        }
    }
}
