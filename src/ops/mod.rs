trait OperationImplementation {
    type Input;
    type Output;
    type ElementType;
    type Initialised;
    type ErrorType;
    fn init(
        self,
        iter: &mut dyn Iterator<Item = Self::ElementType>,
    ) -> Result<Self::Initialised, Self::ErrorType>;
}
