//! This module will contain the traits and structures for
//! the various individual operations in a neural network, as
//! well as providing methods to chain them together at this
//! lowest level.

mod backward;
mod forward;
mod initialised;
mod trainable;
mod uninitialised;

pub use backward::Operation as BackwardOperation;
pub use forward::Construct;
pub use forward::Operation as ForwardOperation;
pub use initialised::Operation as InitialisedOperation;
pub use trainable::Operation as TrainableOperation;
pub use uninitialised::Operation as UninitialisedOperation;

pub use uninitialised::linear::Linear;
