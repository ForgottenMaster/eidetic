use crate::Optimiser;

/// This trait is used to define a specific operation within
/// the neural network.
///
/// Where possible the Operation trait will hide function calls
/// within the OperationPrivate trait in order to restrict their
/// access to this API.
///
/// This allows us to enforce correct usage of the API in client code
/// by having to go through the correct flow of control.
pub trait Operation: OperationPrivate {}

pub trait OperationPrivate: Sized {
    type Element;
    type InitialisationError;
    type Parameter;
    type ParameterGradient;
    type ParameterIter;
    type Input;
    type Output;
    type PredictionError;

    fn neurons(&self) -> usize;
    fn parameter_iter(&self, parameter: Self::Parameter) -> Self::ParameterIter;
    fn predict(
        &self,
        input: Self::Input,
        parameter: Self::Parameter,
    ) -> Result<Self::Output, Self::PredictionError>;
    fn with_iter_internal(
        self,
        iter: &mut impl Iterator<Item = Self::Element>,
        input_neurons: usize,
    ) -> Result<OperationInitialised<Self>, Self::InitialisationError>;
    fn with_seed_internal(
        self,
        seed: u64,
        input_neurons: usize,
    ) -> Result<OperationInitialised<Self>, Self::InitialisationError>;
}

/// An extension trait for implementations of Operation that adds
/// additional functionality on top of the implementation provided by
/// the Operation trait itself.
pub trait OperationExt: Operation {
    /// This function should be called in order to initialise the parameters
    /// of the operation from the given iterator of elements, and puts the
    /// operation in an initialised state. Used for when we are reconstructing
    /// a neural network from previously stored weights.
    fn with_iter(
        self,
        iter: &mut impl Iterator<Item = Self::Element>,
    ) -> Result<OperationInitialised<Self>, Self::InitialisationError> {
        self.with_iter_internal(iter, 0)
    }

    /// This function should be called in order to initialise the parameters
    /// of the operation randomly from a seed.
    fn with_seed(self, seed: u64) -> Result<OperationInitialised<Self>, Self::InitialisationError> {
        self.with_seed_internal(seed, 0)
    }
}

impl<T: Operation> OperationExt for T {}

/// This structure is used to represent an operation that has been correctly
/// initialised.
///
/// Since the internals are not publicly visible, the only way to construct one
/// of these instances is to call the initialise function on an Operation, thus
/// guaranteeing that the parameter has been correctly initialised.
pub struct OperationInitialised<T: OperationPrivate> {
    parameter: T::Parameter,
    operation: T,
}

impl<T: Operation> OperationInitialised<T>
where
    T::Parameter: Clone,
{
    /// This function can be called to get a *copy* of the elements inside the
    /// operation as an iterator.
    ///
    /// We are required to return and iterator of copies because we would need
    /// GATs (Generic Associated Types) in order to allow the associated type
    /// ParameterIter to be generic over the lifetime of the borrow to self.
    pub fn parameters(&self) -> T::ParameterIter {
        self.operation.parameter_iter(self.parameter.clone())
    }

    /// This function is called in order to predict the output of the operation
    /// given the input, through inference by just using the trained weights.
    pub fn predict(&self, input: T::Input) -> Result<T::Output, T::PredictionError> {
        self.operation.predict(input, self.parameter.clone())
    }

    /// This function takes a given optimiser that operates on parameters and
    /// parameter gradients of the same kind as this operation and progresses the
    /// typestate to the trainable state.
    pub fn with_optimiser<U: Optimiser<T::Parameter, T::ParameterGradient>>(
        self,
        optimiser: U,
    ) -> OperationTrainable<T, U> {
        OperationTrainable {
            initialised: self,
            _optimiser: optimiser,
        }
    }
}

pub struct OperationTrainable<T: Operation, U: Optimiser<T::Parameter, T::ParameterGradient>> {
    initialised: OperationInitialised<T>,
    _optimiser: U,
}

impl<T: Operation, U: Optimiser<T::Parameter, T::ParameterGradient>> OperationTrainable<T, U> {
    pub fn into_initialised(self) -> OperationInitialised<T> {
        self.initialised
    }
}

/*
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
*/
