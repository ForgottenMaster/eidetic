//! This module contains re-exports of only those operations that
//! are considered to be the layers of a neural network. Layers are
//! the level of unit that clients will generally compose together into
//! networks.

pub use crate::operations::uninitialised::composite::Chain;
pub use crate::operations::uninitialised::composite::Operation as Composite;
pub use crate::operations::uninitialised::dense::Operation as Dense;
pub use crate::operations::uninitialised::dropout::Operation as Dropout;
pub use crate::operations::uninitialised::input::Operation as Input;
