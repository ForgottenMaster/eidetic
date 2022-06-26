//! This submodule contains the traits and structures for operations in the
//! uninitialised state. These are operations that will accept and iterator
//! or random seed and will generate the correct size parameter for the operation.

pub mod linear;

use crate::private::Sealed;
use crate::Result;

/// This trait is used to represent an operation in an uninitialised state
/// that must be initialised before it's used. These operations can be
/// initialised with either an iterator of elements or a random seed.
pub trait Operation: Sealed + Sized {
    /// The type of the underlying element that's used to represent
    /// the data.
    type Element;

    /// This is a type representing the next state in the typestate sequence
    /// which is an initialised operation with generated parameter, etc.
    type Initialised;

    /// This function can be called to initialise the parameters of the operation
    /// from an iterator that yields elements of the expected type for the operation.
    /// Returns an initialised version of the operation if successful. Also returns the number
    /// of output neurons from the operation.
    ///
    /// # Errors
    /// `Error` if the initialisation fails (likely due to invalid count of elements provided).
    fn with_iter(
        self,
        mut iter: impl Iterator<Item = Self::Element>,
    ) -> Result<(Self::Initialised, usize)> {
        self.with_iter_private(&mut iter, 0)
    }

    /// This function can be called to initialise the parameters of the operation
    /// randomly using the given RNG seed. Returns the initialised version of the operation
    /// if successful. Also returns the number of output neurons from the operation.
    ///
    /// # Errors
    /// `Error` if the initialisation fails (likely due to invalid count of elements provided).
    fn with_seed(self, seed: u64) -> Result<(Self::Initialised, usize)> {
        self.with_seed_private(seed, 0)
    }

    #[doc(hidden)]
    fn with_iter_private(
        self,
        iter: &mut impl Iterator<Item = Self::Element>,
        input_neuron_count: usize,
    ) -> Result<(Self::Initialised, usize)>;

    #[doc(hidden)]
    fn with_seed_private(
        self,
        seed: u64,
        input_neuron_count: usize,
    ) -> Result<(Self::Initialised, usize)>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::uninitialised;

    struct StubOperationUninitialised(usize);
    impl Sealed for StubOperationUninitialised {}
    impl uninitialised::Operation for StubOperationUninitialised {
        type Element = ();
        type Initialised = StubOperationInitialised;
        fn with_iter_private(
            self,
            iter: &mut impl Iterator<Item = Self::Element>,
            _input_neuron_count: usize,
        ) -> Result<(Self::Initialised, usize)> {
            Ok((StubOperationInitialised(iter.count()), self.0))
        }
        fn with_seed_private(
            self,
            seed: u64,
            _input_neuron_count: usize,
        ) -> Result<(Self::Initialised, usize)> {
            Ok((StubOperationInitialised(seed as usize), self.0))
        }
    }

    #[derive(Debug, PartialEq)]
    struct StubOperationInitialised(usize);

    #[test]
    fn test_operation_initialisation_with_iter() {
        // Arrange
        let uninit = StubOperationUninitialised(42);
        let array = [(); 7];

        // Act
        let (init, neurons) = uninit.with_iter(array.into_iter()).unwrap();

        // Assert
        assert_eq!(init, StubOperationInitialised(7));
        assert_eq!(neurons, 42);
    }

    #[test]
    fn test_operation_initialisation_with_seed() {
        // Arrange
        let uninit = StubOperationUninitialised(112);
        let seed = 42;

        // Act
        let (init, neurons) = uninit.with_seed(seed).unwrap();

        // Assert
        assert_eq!(init, StubOperationInitialised(42));
        assert_eq!(neurons, 112);
    }
}
