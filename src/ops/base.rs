use crate::sealed::Sealed;

/// A trait which identifies the required information to act as an operation within
/// a neural network that is currently in the unininitialised state.
///
/// This is the state that operations will be in when the network is first constructed
/// and in order to be mapped to an initialised state will need to be provided with a
/// stream of elements from which to build the initial parameters.
///
/// Unfortunately due to the way we're using typestate for making use of the network
/// I couldn't find a good way of branching the typestate mapping from OperationInitialised
/// to OperationTrainable based on the type of optimiser we're using.... therefore we have
/// to encode the optimiser we're using in at the type level here and pass it through up to
/// the OperationTrainable state.
///
/// Note that since this is a subtrait of the Sealed trait which is not visible
/// in the public API, this trait can only be implemented by types within the crate.
///
/// # Parameters
/// T - The type of the optimiser factory we want to use when we put the operation into a trainable state
pub trait OperationUninitialised<T>: Sealed {
    /// The type of elements that the operation uses that will be the type of the
    /// items in the iterator provided for initialisation.
    type ElementType;

    /// The type of an object that encodes this operation in an initialised and ready to
    /// use state.
    type Initialised;

    /// The type of the error variant for when initialisation fails for some reason.
    type Error;

    /// This function can be called with an iterator of elements to construct the
    /// parameters from and will either return a valid object representing the initialised state of this operation,
    /// or an error.
    fn initialise(
        self,
        iter: &mut impl Iterator<Item = Self::ElementType>,
    ) -> Result<Self::Initialised, Self::Error>;
}

/// This trait defines the valid functionality for an operation that has been
/// correctly initialised and can be used for inference or prepared for training.
///
/// Note that since this is a subtrait of the Sealed trait which is not visible
/// in the public API, this trait can only be implemented by types within the crate.
///
/// # Parameters
/// T - The factory for the type of optimiser we will be using which we will use to create the optimiser when moving into the trainable state
pub trait OperationInitialised<T>: Sealed {
    /// The type of elements inside the parameters/inputs of the operation.
    type ElementType;

    /// The type of the input provided to the operation during the forward pass.
    type Input;

    /// The type of the output produced by the operation.
    type Output;

    /// The type of which an instance can be used to train this operation, has the optimiser baked in.
    type Trainable;

    /// The type of the error variant when trying to perform inference or training.
    type Error;

    /// Call this function if we want to perform inference with a given input with the
    /// current parameter, returns the output if inference was successful, or an error
    /// otherwise. Note that we only require an immutable reference here as we're not
    /// modifying any weights.
    fn predict(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;

    /// Called in order to gain access to the flattened stream of parameters for this operation
    /// through an iterator only valid during the callback that was passed. This can be used to
    /// gain access to the parameters on a trained network for example to store off to disk for
    /// later reconstruction.
    fn parameters(&self, func: impl FnMut(&mut dyn Iterator<Item = Self::ElementType>));

    /// Makes the operation trainable which will then create the optimiser from the factory that
    /// will be used during the optimisation step.
    fn into_trainable(self) -> Self::Trainable;
}

/// This trait defines the valid functionality for an operation that has been put into a trainable
/// state and is ready for the forward/backprop passes to start.
///
/// Note that since this is a subtrait of the Sealed trait which is not visible in the
/// public API, this trait can only be implemented by types within the crate.
pub trait OperationTrainable: Sealed {
    /// The type of an object that encodes this operation after having the forward pass run on it.
    type Forward;

    /// The type of the operation that is in an initialised but not trainable state.
    /// Used to go back after training to access the trained parameters for writing to file, etc.
    type Initialised;

    /// This function can be called in order to run a forward pass on the trainable
    /// operation. It takes the trainable operation by mutable reference since the API then
    /// means we can just drop or apply the training from that pass and we don't take ownership
    /// of this state which we will reuse.
    fn forward(&mut self) -> Self::Forward;

    /// Puts this trainable operation back into an initialised state which removes all the optimisers
    /// and allows access to the trained parameters again.
    fn into_initialised(self) -> Self::Initialised;
}

/// This trait defines the valid functionality for an operation that has had the forward pass ran
/// on it and is ready for the backpropagation process to start.
///
/// Note that since this is a subtrait of the Sealed trait which is not visible in the
/// public API, this trait can only be implemented by types within the crate.
pub trait OperationForward: Sealed {
    /// The type of the output gradient provided to the operation during the backward pass.
    type OutputGradient;

    /// The type encoding the state of this operation after having backprop successfully run on it.
    type Backprop;

    /// The type of the error that this operation produces if backpropagation somehow fails.
    type Error;

    /// This function is called to progress the operation into the backpropagation state.
    /// It consumes self since we can only do backprop once per forward pass, before we either
    /// drop and don't apply any changes, or we use an optimiser to apply the gradients.
    fn backward(self, output_gradient: Self::OutputGradient)
        -> Result<Self::Backprop, Self::Error>;
}

/// This trait defines the functionality for an operation that has had the backpropagation run on it
/// and now can be applied using an optimiser.
///
/// Note that since this is a subtrait of the Sealed trait which is not visible in the public API, this
/// trait can only be implemented by types within the crate.
pub trait OperationBackward: Sealed {
    /// Invoked to finalise the backpropagation by applying the calculated
    /// gradients to parameters using the built in optimiser that was baked
    /// in during the prepare step.
    fn optimise(self);
}
