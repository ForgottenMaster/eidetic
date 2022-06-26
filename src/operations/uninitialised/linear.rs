use crate::operations::uninitialised;
use crate::private::Sealed;
use crate::Result;
use core::marker::PhantomData;

/// This operation represents a linear activation function on
/// a rank 2 tensor.
pub struct Linear<T>(PhantomData<T>, usize);

impl<T> Linear<T> {
    /// Function that constructs a new Linear operation given the
    /// expected number of neurons (columns) of the data passing through
    /// it.
    #[must_use]
    pub const fn new(neuron_count: usize) -> Self {
        Self(PhantomData, neuron_count)
    }
}

impl<T> Sealed for Linear<T> {}
impl<T> uninitialised::Operation for Linear<T> {
    type Element = T;
    type Initialised = (); // TODO: replace this with something sensible

    fn output_neuron_count(&self) -> usize {
        self.1
    }

    fn with_iter_private(
        self,
        _iter: &mut impl Iterator<Item = Self::Element>,
        _input_neuron_count: usize,
    ) -> Result<Self::Initialised> {
        Ok(()) // TODO: replace this with something sensible
    }

    fn with_seed_private(
        self,
        _seed: u64,
        _input_neuron_count: usize,
    ) -> Result<Self::Initialised> {
        Ok(()) // TODO: replace this with something sensible
    }
}
