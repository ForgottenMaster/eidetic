//! This module will contain all the handlers for tracking and updating the
//! learning rate for use with optimisers such as SGD.

mod fixed;

use crate::private::Sealed;
use crate::ElementType;

pub use fixed::LearningRateHandler as FixedLearningRateHandler;

/// This trait defines the functionality for a type to be used
/// in optimisation to handle and provide the learning rate. Is able
/// to be initialised at the beginning of training, report the current
/// learning rate, and perform some logic at the end of an epoch.
/// Note that like all traits in the library, this trait is sealed so cannot be implemented by foreign types.
pub trait LearningRateHandler: Sealed {
    /// Provides the current value of the learning rate to the
    /// optimiser when asked.
    fn learning_rate(&self) -> ElementType;

    /// Called at the beginning of training with the number of epochs
    /// we will be running over. Can be used to determine the increments
    /// for learning rate update each epoch.
    fn init(&mut self, epochs: u32);

    /// Called at the end of every epoch and provides an opportunity to update
    /// the learning rate for next time.
    fn end_epoch(&mut self);
}
