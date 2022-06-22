use crate::sealed::Sealed;

/// This trait represents an optimiser during neural network training which
/// is responsible for updating parameters with gradients.
pub trait Optimiser: Sealed {}
