use crate::activations::ActivationFunction;
use crate::operations::initialised;
use crate::operations::UninitialisedOperation;
use crate::private::Sealed;
use crate::ElementType;
use crate::Result;

/// This is a sigmoidal activation function which is a nonlinear
/// activation function using the sigmoid function.
#[derive(Default)]
pub struct Operation(());

impl Operation {
    /// This function is used to construct a new Sigmoid activation
    /// to be passed in to a dense layer within a network.
    #[must_use]
    pub const fn new() -> Self {
        Self(())
    }
}

impl Sealed for Operation {}
impl ActivationFunction for Operation {}
impl UninitialisedOperation for Operation {
    type Initialised = initialised::sigmoid::Operation;

    fn with_iter_private(
        self,
        _iter: &mut impl Iterator<Item = ElementType>,
        input_neuron_count: usize,
    ) -> Result<(Self::Initialised, usize)> {
        Ok((
            initialised::sigmoid::Operation {
                neurons: input_neuron_count,
            },
            input_neuron_count,
        ))
    }

    fn with_seed_private(
        self,
        _seed: u64,
        input_neuron_count: usize,
    ) -> (Self::Initialised, usize) {
        (
            initialised::sigmoid::Operation {
                neurons: input_neuron_count,
            },
            input_neuron_count,
        )
    }
}
