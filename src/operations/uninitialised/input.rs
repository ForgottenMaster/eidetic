use crate::operations::initialised;
use crate::operations::UninitialisedOperation;
use crate::optimisers::null;
use crate::private::Sealed;
use crate::ElementType;
use crate::Result;
use core::marker::PhantomData;

/// This structure represents an input operation which will be used as the very first
/// operation in a sequence. This is to ensure that the neuron count is allowed to be defined
/// for the output if the input layer (number of columns in the output), but that the neuron
/// count is allowed to not be specified for the input. This is different from the Linear activation
/// function for example where the output neuron count is the same as the input - hence they need to be
/// two different functions.
pub struct Input {
    neuron_count: usize,
}

impl Input {
    /// This function is used to construct a new Input operation with a given
    /// neuron count. If data is provided when running/training the network and the
    /// neuron/column count doesn't match then this will be an error.
    #[must_use]
    pub const fn new(neuron_count: usize) -> Self {
        Self { neuron_count }
    }
}

impl Sealed for Input {}
impl UninitialisedOperation for Input {
    type Initialised = initialised::input::Operation<null::OptimiserFactory>;

    fn with_iter_private(
        self,
        _iter: &mut impl Iterator<Item = ElementType>,
        _input_neuron_count: usize,
    ) -> Result<(Self::Initialised, usize)> {
        Ok((
            initialised::input::Operation {
                neurons: self.neuron_count,
                phantom_data: PhantomData,
            },
            self.neuron_count,
        ))
    }

    fn with_seed_private(
        self,
        _seed: u64,
        _input_neuron_count: usize,
    ) -> (Self::Initialised, usize) {
        (
            initialised::input::Operation {
                neurons: self.neuron_count,
                phantom_data: PhantomData,
            },
            self.neuron_count,
        )
    }
}
