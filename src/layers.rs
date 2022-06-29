//! This module contains re-exports of only those operations that
//! are considered to be the layers of a neural network. Layers are
//! the level of unit that clients will generally compose together into
//! networks.

pub use crate::operations::uninitialised::input::Operation as Input;

/// This marker trait is used to identify those operations that are
/// considered layers which will then be chainable.
pub trait Layer: crate::operations::UninitialisedOperation {}
