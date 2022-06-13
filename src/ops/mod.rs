use crate::tensor::InvalidTensorSizeError;

trait OperationImplementation {
    type Input;
    type Output;
    type ElementType;
    type Initialised;
    fn init(
        self,
        iter: &mut dyn Iterator<Item = Self::ElementType>,
    ) -> Result<Self::Initialised, InvalidTensorSizeError>;
}
