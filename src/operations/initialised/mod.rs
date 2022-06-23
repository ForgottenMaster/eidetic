//! This submodule contains the traits and structures for operations in the
//! initialised state.

use crate::operations::trainable::{OperationTrainable, Trainable};
use crate::optimisers::OptimiserFactory;
use crate::private::Sealed;

/// This trait is used to represent an operation in an initialised state that has a valid
/// parameter stored internally, and which can be used to run inference or prepared for
/// training by providing an optimiser.
pub trait OperationInitialised: Sealed + Sized {
    /// The type of the elements within the operation.
    /// Used to ensure that the ParameterIter item type matches.
    type Element;

    /// The type that is passed into the operation.
    type Input;

    /// The type that is output from the operation.
    type Output;

    /// The type of the parameter stored within the operation.
    type Parameter;

    /// The iterator type that will be returned when asked for that
    /// iterates over the elements of the (flattened) parameter(s) within
    /// this operation (for example, for serialization/saving after training).
    type ParameterIter: Iterator<Item = Self::Element>;

    /// The error type that will be emitted if prediction fails such as if the
    /// input type has an incorrect shape for example.
    type Error;

    /// This is the type that is used once the operation is placed in a trainable
    /// state, and provides the forward pass functionality.
    type Trainable: OperationTrainable<Initialised = Self>;

    /// This function can be called to get an iterator over the copies of the elements
    /// stored within this operation's parameter. The parameter is flattened to a single
    /// stream for emitting. This is guaranteed to be the same order as is accepted by the
    /// with_iter initialisation function for networks.
    fn iter(&self) -> Self::ParameterIter;

    /// This function can take a given input and run it through the operation/network to produce
    /// the output for it. Can produce an error if (for example) the input is an incorrect shape.
    fn predict(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;

    /// This function can be used to prepare the initialised network/operation for training by
    /// specifying a specific optimiser. Since this operation might be a chain, and since we want
    /// each element of the chain to get its own optimiser, we need to accept the optimiser as a factory
    /// which can produce a specific optimiser for ourselves.
    fn with_optimiser<T: OptimiserFactory<Self>>(
        self,
        factory: T,
    ) -> Trainable<Self::Trainable, T::Optimiser>;
}
