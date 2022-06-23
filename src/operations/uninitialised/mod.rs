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
    type ElementType;

    /// This is a type representing the next state in the typestate sequence
    /// which is an initialised operation with generated parameter, etc.
    type Initialised: OperationInitialised;

    /// This returns the output neuron count for the operation.
    /// We represent this using the element type because it could be
    /// used with xavier initialisation
    fn output_neuron_count(&self) -> Self::ElementType;

    /// This function can be called to initialise the parameters of the operation
    /// from an iterator that yields elements of the expected type for the operation.
    fn with_iter(self, mut iter: impl Iterator<Item = Self::ElementType>) -> Self::Initialised {
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
        iter: &mut impl Iterator<Item = Self::ElementType>,
        input_neuron_count: Self::ElementType,
    ) -> Self::Initialised;

    #[doc(hidden)]
    fn with_seed_private(
        self,
        seed: u64,
        input_neuron_count: Self::ElementType,
    ) -> Self::Initialised;
}
