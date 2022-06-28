use crate::operations::forward;
use crate::operations::trainable::input::Operation;

impl<'a> forward::Construct<'a> for Operation {
    type Forward = Forward<'a>;
    fn construct(&'a mut self) -> Self::Forward {
        Forward::<'a>(self)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Forward<'a>(&'a mut Operation);
