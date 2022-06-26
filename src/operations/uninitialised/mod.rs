//! This submodule contains the traits and structures for operations in the
//! uninitialised state. These are operations that will accept and iterator
//! or random seed and will generate the correct size parameter for the operation.

pub mod input;

use crate::private::Sealed;
use crate::ElementType;
use crate::Result;

/// This trait is used to represent an operation in an uninitialised state
/// that must be initialised before it's used. These operations can be
/// initialised with either an iterator of elements or a random seed.
pub trait Operation: Sealed + Sized {
    /// This is a type representing the next state in the typestate sequence
    /// which is an initialised operation with generated parameter, etc.
    type Initialised;

    /// This function can be called to initialise the parameters of the operation
    /// from an iterator that yields elements of the expected type for the operation.
    /// Returns an initialised version of the operation if successful. Also returns the number
    /// of output neurons from the operation.
    ///
    /// # Errors
    /// `Error` if the initialisation fails due to the incorrect number of elements being provided.
    fn with_iter(self, mut iter: impl Iterator<Item = ElementType>) -> Result<Self::Initialised> {
        let (initialised, _) = self.with_iter_private(&mut iter, 0)?;
        Ok(initialised)
    }

    /// This function is called to initialise the parameters of the operation
    /// from a random seed. This is used when the network isn't already trained
    /// and is being constructed for the first time.
    fn with_seed(self, seed: u64) -> Self::Initialised {
        self.with_seed_private(seed, 0).0
    }

    #[doc(hidden)]
    fn with_iter_private(
        self,
        iter: &mut impl Iterator<Item = ElementType>,
        input_neuron_count: usize,
    ) -> Result<(Self::Initialised, usize)>;

    #[doc(hidden)]
    fn with_seed_private(self, seed: u64, input_neuron_count: usize) -> (Self::Initialised, usize);
}
