//! This submodule contains the traits and structures for operations in the
//! uninitialised state. These are operations that will accept and iterator
//! or random seed and will generate the correct size parameter for the operation.

use crate::operations::initialised::OperationInitialised;
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

    /// This is a type representing the next state in the typestate sequence
    /// which is an initialised operation with generated parameter, etc.
    type Initialised: OperationInitialised;

    /// This returns the output neuron count for the operation.
    fn output_neuron_count(&self) -> usize;

    /// This function can be called to initialise the parameters of the operation
    /// from an iterator that yields elements of the expected type for the operation.
    fn with_iter(self, mut iter: impl Iterator<Item = Self::Element>) -> Self::Initialised {
        let input_neuron_count = self.output_neuron_count();
        self.with_iter_private(&mut iter, input_neuron_count)
    }

    /// This function can be called to initialise the parameters of the operation
    /// randomly using the given RNG seed.
    fn with_seed(self, seed: u64) -> Self::Initialised {
        let input_neuron_count = self.output_neuron_count();
        self.with_seed_private(seed, input_neuron_count)
    }

    #[doc(hidden)]
    fn with_iter_private(
        self,
        iter: &mut impl Iterator<Item = Self::Element>,
        input_neuron_count: usize,
    ) -> Self::Initialised;

    #[doc(hidden)]
    fn with_seed_private(self, seed: u64, input_neuron_count: usize) -> Self::Initialised;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubOperationUninitialised(usize);
    impl Sealed for StubOperationUninitialised {}
    impl OperationUninitialised for StubOperationUninitialised {
        type Element = ();
        type Input = ();
        type Output = ();
        type Initialised = StubOperationInitialised;
        fn output_neuron_count(&self) -> usize {
            self.0
        }
        fn with_iter_private(
            self,
            iter: &mut impl Iterator<Item = Self::Element>,
            input_neuron_count: usize,
        ) -> Self::Initialised {
            StubOperationInitialised(input_neuron_count, self.output_neuron_count(), iter.count())
        }
        fn with_seed_private(self, seed: u64, input_neuron_count: usize) -> Self::Initialised {
            StubOperationInitialised(
                input_neuron_count,
                self.output_neuron_count(),
                seed as usize,
            )
        }
    }

    #[derive(Debug, PartialEq)]
    struct StubOperationInitialised(usize, usize, usize);
    impl Sealed for StubOperationInitialised {}
    impl OperationInitialised for StubOperationInitialised {}

    #[test]
    fn test_operation_initialisation_with_iter() {
        // Arrange
        let uninit = StubOperationUninitialised(42);
        let array = [(); 7];

        // Act
        let init = uninit.with_iter(array.into_iter());

        // Assert
        assert_eq!(init, StubOperationInitialised(42, 42, 7));
    }

    #[test]
    fn test_operation_initialisation_with_seed() {
        // Arrange
        let uninit = StubOperationUninitialised(112);
        let seed = 42;

        // Act
        let init = uninit.with_seed(seed);

        // Assert
        assert_eq!(init, StubOperationInitialised(112, 112, 42));
    }
}
