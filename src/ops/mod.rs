use crate::error::InvalidSizeError;
use crate::marker::Sealed;

/// This trait represents the functionality of a particular operation
/// within a neural network. It can either be a fundamental operation
/// such as a weighted sum or bias addition, or could be some kind of compound
/// operation like a layer or an adapter for an existing operation.
///
/// This is a sealed trait because it's intended that we will be adding
/// sufficient functionality that we don't need the flexibility of external implementations
/// and additionally, since it's sealed we are able to add methods in future without breaking
/// any crates using this API.
pub trait OperationImplementation: Sealed {
    /// The type of elements that the operation operates on (the type of elements in the tensor).
    type ElementType;

    /// The type of the input to the operation.
    /// Note that input gradients are assumed to be the same type.
    type Input;

    /// The type of the output to the operation.
    /// Note that output gradients are assumed to be the same type.
    type Output;

    /// This is a concrete type which represents the initialised
    /// state of the operation in the initialized neural network.
    /// This is the type that will be used to perform the operation
    /// when running or training the network.
    type Initialised;

    /// Initialises the operation's parameters with the given iterator over elements of the
    /// type T.
    ///
    /// This could come from either random number generation, or from a fixed sequence of elements
    /// such as might have been serialized to disk.
    ///
    /// If the provided stream of elements does not have the correct number to initialise the
    /// operation then an Error is returned, otherwise the initialised operation is returned.
    fn init(
        self,
        iter: &mut dyn Iterator<Item = Self::ElementType>,
    ) -> Result<Self::Initialised, InvalidSizeError>;
}
