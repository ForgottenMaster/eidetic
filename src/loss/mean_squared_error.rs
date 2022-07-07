use crate::loss::Loss;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Error, Result};

/// This structure defines the "Mean Squared Error" loss function.
pub struct MeanSquaredError(());

impl MeanSquaredError {
    /// Constructs a new instance of the `MeanSquaredError` loss
    /// function.
    #[must_use]
    pub const fn new() -> Self {
        Self(())
    }
}

impl Loss for MeanSquaredError {
    fn loss(
        &self,
        predictions: &Tensor<rank::Two>,
        targets: &Tensor<rank::Two>,
    ) -> Result<(ElementType, Tensor<rank::Two>)> {
        let (predictions, targets) = (&predictions.0, &targets.0);
        let predictions_dim = predictions.raw_dim();
        let targets_dim = targets.raw_dim();
        if predictions_dim == targets_dim {
            // Get the error first (squared error sum).
            let error = predictions - targets;
            let squared_error = &error * &error;
            let squared_error_sum = squared_error.sum();
            let count = u16::try_from(predictions.nrows()).map_err(|_| Error(()))?;
            let count: ElementType = count.into();
            let squared_error_sum = squared_error_sum / count;

            // Calculate the output gradient/loss gradient.
            let average_error = error / count;
            let average_error = average_error * 2.0;
            let average_error = Tensor(average_error);

            // Return both.
            Ok((squared_error_sum, average_error))
        } else {
            Err(Error(()))
        }
    }
}
impl Sealed for MeanSquaredError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_success() {
        // Arrange
        let mse = MeanSquaredError::new();
        let predictions = Tensor::<rank::Two>::new((3, 1), [23.0, -17.0, 22.0]).unwrap();
        let targets = Tensor::<rank::Two>::new((3, 1), [12.0, 13.0, -7.0]).unwrap();
        let (expected_loss, expected_output_gradient) = {
            (
                620.6666666666666,
                Tensor::<rank::Two>::new((3, 1), [7.333333333333333, -20.0, 19.33333333333333333])
                    .unwrap(),
            )
        };

        // Act
        let (loss, output_gradient) = mse.loss(&predictions, &targets).unwrap();

        // Assert
        assert_eq!(loss, expected_loss);
        assert_eq!(output_gradient, expected_output_gradient);
    }
}
