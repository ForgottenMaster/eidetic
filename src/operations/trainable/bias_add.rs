use crate::operations::initialised;

#[derive(Debug, PartialEq)]
pub struct Operation<T> {
    pub(crate) _optimiser: T,
    pub(crate) _initialised: initialised::bias_add::Operation,
}
