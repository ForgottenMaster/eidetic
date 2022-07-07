//! This module contains the Loss trait defining a loss function to be used
//! to calculate the initial gradient for the backward pass, along with the
//! various loss functions we can use.

mod mean_squared_error;

pub use mean_squared_error::MeanSquaredError;

use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Result};

/// This trait defines a loss function that can be used to calculate loss
/// and the loss gradient for training a neural network.
pub trait Loss: Sealed {
    /// Calculates the loss given predictions, along with the associated targets.
    /// If the shapes don't match then an error is returned, otherwise returns the
    /// loss value, along with the loss gradient tensor.
    ///
    /// # Errors
    /// Returns an error if the predictions and targets don't have the same shape.
    fn loss(
        &self,
        predictions: &Tensor<rank::Two>,
        targets: &Tensor<rank::Two>,
    ) -> Result<(ElementType, Tensor<rank::Two>)>;
}
