use crate::operations::{forward, BackwardOperation};
use crate::private::Sealed;

#[derive(Debug, PartialEq)]
pub struct Operation<'a> {
    pub(crate) _forward: forward::dropout::Operation<'a>,
}

impl<'a> Sealed for Operation<'a> {}

impl<'a> BackwardOperation for Operation<'a> {
    fn optimise(self) {}
}
