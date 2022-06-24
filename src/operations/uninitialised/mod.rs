//! This submodule contains the traits and structures for operations in the
//! uninitialised state. These are operations that will accept and iterator
//! or random seed and will generate the correct size parameter for the operation.

use crate::operations::initialised::OperationInitialised;
use crate::optimisers::NullOptimiser;
use crate::private::Sealed;

/// This trait is used to represent an operation in an uninitialised state
/// that must be initialised before it's used. These operations can be
/// initialised with either an iterator of elements or a random seed.
pub trait OperationUninitialised: Sealed + Sized {
    /// The type of the underlying element that's used to represent
    /// the data.
    type Element;

    /// The input type that the operation is expecting to receive.
    type Input;

    /// The output type that the operation will produce.
    type Output;

    /// This is the type that's used in order to report an error
    /// with initialisation (e.g. not enough elements in the iterator, etc.).
    type Error;

    /// This is a type representing the next state in the typestate sequence
    /// which is an initialised operation with generated parameter, etc.
    type Initialised: OperationInitialised<NullOptimiser>;

    /// This returns the output neuron count for the operation.
    fn output_neuron_count(&self) -> usize;

    /// This function can be called to initialise the parameters of the operation
    /// from an iterator that yields elements of the expected type for the operation.
    fn with_iter(
        self,
        mut iter: impl Iterator<Item = Self::Element>,
    ) -> Result<Self::Initialised, Self::Error> {
        let input_neuron_count = self.output_neuron_count();
        self.with_iter_private(&mut iter, input_neuron_count)
    }

    /// This function can be called to initialise the parameters of the operation
    /// randomly using the given RNG seed.
    fn with_seed(self, seed: u64) -> Result<Self::Initialised, Self::Error> {
        let input_neuron_count = self.output_neuron_count();
        self.with_seed_private(seed, input_neuron_count)
    }

    #[doc(hidden)]
    fn with_iter_private(
        self,
        iter: &mut impl Iterator<Item = Self::Element>,
        input_neuron_count: usize,
    ) -> Result<Self::Initialised, Self::Error>;

    #[doc(hidden)]
    fn with_seed_private(
        self,
        seed: u64,
        input_neuron_count: usize,
    ) -> Result<Self::Initialised, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::trainable::OperationTrainable;
    use crate::optimisers::OptimiserFactory;

    struct StubOperationUninitialised(usize);
    impl Sealed for StubOperationUninitialised {}
    impl OperationUninitialised for StubOperationUninitialised {
        type Element = ();
        type Input = ();
        type Output = ();
        type Error = ();
        type Initialised = StubOperationInitialised;
        fn output_neuron_count(&self) -> usize {
            self.0
        }
        fn with_iter_private(
            self,
            iter: &mut impl Iterator<Item = Self::Element>,
            input_neuron_count: usize,
        ) -> Result<Self::Initialised, Self::Error> {
            Ok(StubOperationInitialised(
                input_neuron_count,
                self.output_neuron_count(),
                iter.count(),
            ))
        }
        fn with_seed_private(
            self,
            seed: u64,
            input_neuron_count: usize,
        ) -> Result<Self::Initialised, Self::Error> {
            Ok(StubOperationInitialised(
                input_neuron_count,
                self.output_neuron_count(),
                seed as usize,
            ))
        }
    }

    #[derive(Debug, PartialEq)]
    struct StubOperationInitialised(usize, usize, usize);
    impl Sealed for StubOperationInitialised {}
    impl<T: OptimiserFactory<()>> OperationInitialised<T> for StubOperationInitialised {
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
        fn with_optimiser(self, _factory: T) -> Self::Trainable {
            unimplemented!()
        }
    }

    struct StubOperationTrainable;
    impl Sealed for StubOperationTrainable {}
    impl<T: OptimiserFactory<()>> OperationTrainable<T> for StubOperationTrainable {
        type Initialised = StubOperationInitialised;
        fn into_initialised(self) -> Self::Initialised {
            unimplemented!()
        }
    }

    #[test]
    fn test_operation_initialisation_with_iter() {
        // Arrange
        let uninit = StubOperationUninitialised(42);
        let array = [(); 7];

        // Act
        let init = uninit.with_iter(array.into_iter()).unwrap();

        // Assert
        assert_eq!(init, StubOperationInitialised(42, 42, 7));
    }

    #[test]
    fn test_operation_initialisation_with_seed() {
        // Arrange
        let uninit = StubOperationUninitialised(112);
        let seed = 42;

        // Act
        let init = uninit.with_seed(seed).unwrap();

        // Assert
        assert_eq!(init, StubOperationInitialised(112, 112, 42));
    }

    // These tests aren't actually testing anything because the system under
    // test at this point is not the initialised stub. However due to code coverage
    // we should execute these functions to remove the lines from uncovered ones.
    #[test]
    #[should_panic]
    fn test_stub_operation_initialised_iter() {
        OperationInitialised::<NullOptimiser>::iter(&StubOperationInitialised(0, 0, 0))
            .next()
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn test_stub_operation_initialised_predict() {
        OperationInitialised::<NullOptimiser>::predict(&StubOperationInitialised(0, 0, 0), ())
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn test_stub_operation_training_into_initialised() {
        OperationTrainable::<NullOptimiser>::into_initialised(StubOperationTrainable);
    }
}
