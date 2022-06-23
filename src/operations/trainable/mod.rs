//! Module containing the traits and types relating
//! to operations and chains of operations in the trainable typestate.

use crate::operations::initialised::OperationInitialised;
use crate::private::Sealed;

/// This trait is implemented on those types that represent
/// an operation that is in a state ready to be trained.
/// This means it has been through the "with_optimiser" function
/// call to bind an optimiser to the network.
pub trait OperationTrainable: Sealed {
    /// This is the type of the initialised version of the operation.
    type Initialised: OperationInitialised;

    /// Calling this function will "go back" from a trainable
    /// state into an initialised one. This allows the trained network
    /// to be used for inference, or allows a different optimiser to be
    /// used (though the optimiser obviously starts from scratch).
    fn into_initialised(self) -> Self::Initialised;
}

/// This struct is used to bind the trainable state of the underlying
/// operation to a specific optimiser inside an opaque instance.
/// This gives us a wrapping type that can orchestrate the optimisation process.
pub struct Trainable<T, U>(T, U);

impl<T, U> Sealed for Trainable<T, U> {}
impl<T: OperationTrainable, U> OperationTrainable for Trainable<T, U> {
    type Initialised = T::Initialised;

    fn into_initialised(self) -> Self::Initialised {
        self.0.into_initialised()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimisers::OptimiserFactory;

    #[derive(Debug, PartialEq)]
    struct StubOperationInitialised(usize);
    impl Sealed for StubOperationInitialised {}
    impl OperationInitialised for StubOperationInitialised {
        type Element = ();
        type Input = ();
        type Output = ();
        type Parameter = ();
        type ParameterIter = core::iter::Empty<()>;
        type Error = ();
        type Trainable = StubOperationTrainable;
        fn iter(&self) -> Self::ParameterIter {
            unimplemented!()
        }
        fn predict(&self, _input: Self::Input) -> Result<Self::Output, Self::Error> {
            unimplemented!()
        }
        fn with_optimiser<T: OptimiserFactory<Self>>(
            self,
            _factory: T,
        ) -> Trainable<Self::Trainable, T::Optimiser> {
            unimplemented!()
        }
    }

    struct StubOperationTrainable(StubOperationInitialised);
    impl Sealed for StubOperationTrainable {}
    impl OperationTrainable for StubOperationTrainable {
        type Initialised = StubOperationInitialised;
        fn into_initialised(self) -> Self::Initialised {
            self.0
        }
    }

    #[test]
    fn test_trainable_into_initialised() {
        // Arrange
        let trainable = Trainable(StubOperationTrainable(StubOperationInitialised(42)), ());

        // Act
        let initialised = trainable.into_initialised();

        // Assert
        assert_eq!(initialised, StubOperationInitialised(42));
    }
}
