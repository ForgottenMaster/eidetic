use crate::operations::BackwardOperation;
use crate::private::Sealed;

pub struct Operation(pub(crate) ());

impl Sealed for Operation {}
impl BackwardOperation for Operation {
    fn optimise(self) {}
}
