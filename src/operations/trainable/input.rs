use crate::operations::forward::Construct;
use crate::operations::{forward, initialised, trainable};
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{Error, Result};

#[derive(Debug, PartialEq)]
pub struct Operation<T>(initialised::input::Operation<T>);

impl<T> Sealed for Operation<T> {}
impl<T> trainable::Operation for Operation<T> {
    type Input = Tensor<rank::Two>;
    type Output = Tensor<rank::Two>;
    type Initialised = initialised::input::Operation<T>;

    fn new(init: Self::Initialised) -> Self {
        Self(init)
    }

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

impl<'a, T: 'a> forward::Construct<'a> for Operation<T> {
    type Forward = Forward<'a, T>;
    fn construct(&'a mut self) -> Self::Forward {
        Forward::<'a>(self)
    }
}

#[derive(Debug, PartialEq)]
pub struct Forward<'a, T>(&'a mut Operation<T>);
