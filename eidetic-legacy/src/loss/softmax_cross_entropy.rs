use crate::loss::Loss;
use crate::private::Sealed;
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Error, Result};
use ndarray::{Array, Axis, Ix2};

/// This is a loss function which is specialised for calculating the loss
/// for classification problems where the outputs should represent probabilities of
/// being in a certain class. If there's only a single feature/column then it will use
pub struct SoftmaxCrossEntropy(());

impl SoftmaxCrossEntropy {
    /// Constructs a new instance of the `SoftmaxCrossEntropy` loss
    /// function which is a good one to use for classification problems
    /// where the output is based on probabilities.
    #[must_use]
    pub const fn new() -> Self {
        Self(())
    }
}

impl Loss for SoftmaxCrossEntropy {
    fn loss(
        &self,
        predictions: &Tensor<rank::Two>,
        targets: &Tensor<rank::Two>,
    ) -> Result<(ElementType, Tensor<rank::Two>)> {
        let (predictions, targets) = (&predictions.0, &targets.0);
        let (predictions_dim, targets_dim) = (predictions.raw_dim(), targets.raw_dim());
        if predictions_dim == targets_dim {
            // Map the single class predictions to multi-class ones by adding a dummy feature.
            let is_single_class = predictions_dim[1] == 1;
            let (predictions, targets) = if is_single_class {
                (
                    single_class_to_dual(predictions),
                    single_class_to_dual(targets),
                )
            } else {
                ((*predictions).clone(), (*targets).clone())
            };

            // calculate the softmaxed predictions.
            let predictions = calculate_softmax_predictions(predictions);

            // calculate the output sum.
            let minuend = targets.mapv(|elem| -elem) * predictions.mapv(ElementType::ln);
            let subtrahend =
                targets.mapv(|elem| 1.0 - elem) * predictions.mapv(|elem| (1.0 - elem).ln());
            let loss = minuend - subtrahend;
            let loss = loss.sum();

            // calculate the input gradient for the backward pass.
            let loss_gradient = predictions - targets;
            let loss_gradient = if is_single_class {
                dual_class_to_single(&loss_gradient)
            } else {
                loss_gradient
            };
            let loss_gradient = Tensor(loss_gradient);

            // done!
            Ok((loss, loss_gradient))
        } else {
            Err(Error(()))
        }
    }
}
impl Sealed for SoftmaxCrossEntropy {}

fn calculate_softmax_predictions(predictions: Array<ElementType, Ix2>) -> Array<ElementType, Ix2> {
    assert_ne!(predictions.ncols(), 1); // shouldn't be called with only a single feature.
    let mut predictions = softmax(predictions);
    predictions.mapv_inplace(|elem| {
        ElementType::clamp(elem, ElementType::EPSILON, 1.0 - ElementType::EPSILON)
    });
    predictions
}

fn softmax(mut arr: Array<ElementType, Ix2>) -> Array<ElementType, Ix2> {
    arr.map_inplace(|elem| *elem = elem.exp());
    let totals = arr
        .map_axis(Axis(1), |row| row.sum())
        .into_shape((arr.nrows(), 1))
        .unwrap();
    arr / totals
}

fn single_class_to_dual(input: &Array<ElementType, Ix2>) -> Array<ElementType, Ix2> {
    assert_eq!(input.ncols(), 1); // just don't call this function if it's not a single class input.
    let rows = input.nrows();
    Array::from_iter(
        input
            .iter()
            .flat_map(|elem| core::iter::once(*elem).chain(core::iter::once(1.0 - *elem))),
    )
    .into_shape((rows, 2))
    .unwrap()
}

fn dual_class_to_single(input: &Array<ElementType, Ix2>) -> Array<ElementType, Ix2> {
    assert_eq!(input.ncols(), 2); // just don't call this function if it's not dual class. It's private so we can control this.
    input.select(Axis(1), &[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_with_three_classes_and_a_single_observation() {
        // Arrange
        let input = Array::from_iter([5.0, 3.0, 2.0].into_iter())
            .into_shape((1, 3))
            .unwrap();
        #[cfg(not(feature = "f32"))]
        let expected = Array::from_iter(
            [
                0.8437947344813395,
                0.11419519938459449,
                0.042010066134066056,
            ]
            .into_iter(),
        )
        .into_shape((1, 3))
        .unwrap();
        #[cfg(feature = "f32")]
        let expected = Array::from_iter([0.8437947, 0.1141952, 0.042010065].into_iter())
            .into_shape((1, 3))
            .unwrap();

        // Act
        let output = softmax(input);

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_softmax_with_three_classes_and_three_observations() {
        // Arrange
        let input = Array::from_iter([1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 8.0, 7.0, 9.0].into_iter())
            .into_shape((3, 3))
            .unwrap();
        #[cfg(not(feature = "f32"))]
        let expected = Array::from_iter(
            [
                0.09003057317038046,
                0.24472847105479767,
                0.6652409557748219,
                0.6652409557748219,
                0.09003057317038045,
                0.24472847105479764,
                0.24472847105479764,
                0.09003057317038045,
                0.6652409557748219,
            ]
            .into_iter(),
        )
        .into_shape((3, 3))
        .unwrap();
        #[cfg(feature = "f32")]
        let expected = Array::from_iter(
            [
                0.09003057,
                0.24472848,
                0.66524094,
                0.66524094,
                0.090030566,
                0.24472846,
                0.24472846,
                0.09003057,
                0.66524094,
            ]
            .into_iter(),
        )
        .into_shape((3, 3))
        .unwrap();

        // Act
        let output = softmax(input);

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    #[should_panic]
    fn test_single_class_to_dual_with_too_few_features() {
        single_class_to_dual(&Array::<ElementType, _>::ones((0, 0)));
    }

    #[test]
    #[should_panic]
    fn test_single_class_to_dual_with_too_many_features() {
        single_class_to_dual(
            &Array::from_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter())
                .into_shape((3, 2))
                .unwrap(),
        );
    }

    #[test]
    fn test_single_class_to_dual_with_a_single_feature() {
        // Arrange
        let input = Array::from_iter([0.5, 0.75, 0.35].into_iter())
            .into_shape((3, 1))
            .unwrap();
        let expected = Array::from_iter([0.5, 0.5, 0.75, 0.25, 0.35, 0.65].into_iter())
            .into_shape((3, 2))
            .unwrap();

        // Act
        let output = single_class_to_dual(&input);

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    #[should_panic]
    fn test_dual_class_to_single_with_too_few_features() {
        dual_class_to_single(
            &Array::from_iter([0.5, 0.25, 0.35].into_iter())
                .into_shape((3, 1))
                .unwrap(),
        );
    }

    #[test]
    #[should_panic]
    fn test_dual_class_to_single_with_too_many_features() {
        dual_class_to_single(
            &Array::from_iter([0.5, 0.25, 0.35, 0.5, 0.25, 0.35, 0.5, 0.25, 0.35].into_iter())
                .into_shape((3, 3))
                .unwrap(),
        );
    }

    #[test]
    fn test_dual_class_to_single_with_two_features() {
        // Arrange
        let input = Array::from_iter([0.5, 0.5, 0.75, 0.25, 0.35, 0.65].into_iter())
            .into_shape((3, 2))
            .unwrap();
        let expected = Array::from_iter([0.5, 0.75, 0.35].into_iter())
            .into_shape((3, 1))
            .unwrap();

        // Act
        let output = dual_class_to_single(&input);

        // Assert
        assert_eq!(output, expected);
    }

    #[test]
    fn test_loss_with_single_class() {
        // Arrange
        let predictions = Tensor::<rank::Two>::new((3, 1), [0.25, 0.75, 0.45]).unwrap();
        let targets = Tensor::<rank::Two>::new((3, 1), [0.0, 1.0, 0.0]).unwrap();
        #[cfg(not(feature = "f32"))]
        let (expected_loss, expected_gradient) = (
            3.185101256867568,
            Tensor::<rank::Two>::new(
                (3, 1),
                [0.3775406687981454, -0.3775406687981454, 0.47502081252106004],
            )
            .unwrap(),
        );
        #[cfg(feature = "f32")]
        let (expected_loss, expected_gradient) = (
            3.185101,
            Tensor::<rank::Two>::new((3, 1), [0.37754065, -0.37754065, 0.47502083]).unwrap(),
        );
        let loss_function = SoftmaxCrossEntropy::new();

        // Act
        let (loss, gradient) = loss_function.loss(&predictions, &targets).unwrap();

        // Assert
        assert_eq!(loss, expected_loss);
        assert_eq!(gradient, expected_gradient);
    }

    #[test]
    fn test_loss_with_multi_class() {
        // Arrange
        let predictions =
            Tensor::<rank::Two>::new((3, 2), [0.25, 0.75, 0.75, 0.25, 0.45, 0.55]).unwrap();
        let targets = Tensor::<rank::Two>::new((3, 2), [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        #[cfg(not(feature = "f32"))]
        let (expected_loss, expected_gradient) = (
            3.185101256867568,
            Tensor::<rank::Two>::new(
                (3, 2),
                [
                    0.3775406687981454,
                    -0.3775406687981454,
                    -0.3775406687981454,
                    0.3775406687981454,
                    0.47502081252106004,
                    -0.47502081252106,
                ],
            )
            .unwrap(),
        );
        #[cfg(feature = "f32")]
        let (expected_loss, expected_gradient) = (
            3.185101,
            Tensor::<rank::Two>::new(
                (3, 2),
                [
                    0.37754065,
                    -0.37754065,
                    -0.37754065,
                    0.37754065,
                    0.47502083,
                    -0.47502083,
                ],
            )
            .unwrap(),
        );
        let loss_function = SoftmaxCrossEntropy::new();

        // Act
        let (loss, gradient) = loss_function.loss(&predictions, &targets).unwrap();

        // Assert
        assert_eq!(loss, expected_loss);
        assert_eq!(gradient, expected_gradient);
    }

    #[test]
    fn test_loss_error() {
        // Arrange
        let predictions = Tensor::<rank::Two>::new((3, 1), [0.25, 0.75, 0.45]).unwrap();
        let targets = Tensor::<rank::Two>::new((3, 2), [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let loss_function = SoftmaxCrossEntropy::new();

        // Act
        let result = loss_function.loss(&predictions, &targets);

        // Assert
        assert!(result.is_err());
    }
}
