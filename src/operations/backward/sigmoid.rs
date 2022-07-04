use crate::operations::BackwardOperation;
use crate::private::Sealed;

#[derive(Debug, Eq, PartialEq)]
pub struct Operation(pub(crate) ());

impl Sealed for Operation {}
impl BackwardOperation for Operation {
    fn optimise(self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimise() {
        // Arrange
        let operation = Operation(());

        // Act
        operation.optimise();
    }
}
