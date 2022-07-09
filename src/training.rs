//! This module contains training related functionality for
//! taking an initialised network and training it over a certain
//! number of epochs with a certain optimisation strategy, etc.

use crate::loss::Loss;
use crate::operations::{
    BackwardOperation, Forward, ForwardOperation, InitialisedOperation, TrainableOperation,
    WithOptimiser,
};
use crate::tensors::{rank, Tensor};
use crate::{ElementType, Error, Result};
use ndarray::{Array, ArrayView, Axis, Ix2};
use ndarray_rand::{RandomExt, SamplingStrategy};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn generate_batches<'a>(
    batch: &'a Array<ElementType, Ix2>,
    targets: &'a Array<ElementType, Ix2>,
    size: usize,
) -> impl Iterator<Item = (Array<ElementType, Ix2>, Array<ElementType, Ix2>)> + 'a {
    batch
        .axis_chunks_iter(Axis(0), size)
        .zip(targets.axis_chunks_iter(Axis(0), size))
        .map(|(view1, view2)| (view1.to_owned(), view2.to_owned()))
}

fn permute_data(
    mut batch: Array<ElementType, Ix2>,
    targets: &Array<ElementType, Ix2>,
    seed: u64,
) -> (Array<ElementType, Ix2>, Array<ElementType, Ix2>) {
    // get dimensions for later use.
    let (batch_row_count, batch_col_count) = (batch.nrows(), batch.ncols());
    let (targets_row_count, _) = (targets.nrows(), targets.ncols());
    assert_eq!(batch_row_count, targets_row_count);

    // construct RNG from provided seed.
    let mut random_generator = StdRng::seed_from_u64(seed);

    // join batch and targets together side by side for row permutation
    batch.append(Axis(1), targets.into()).unwrap();

    // permute the rows of the axis, don't re-use the indices though as we want
    // to juggle them around.
    let shuffled = batch.sample_axis_using(
        Axis(0),
        batch_row_count,
        SamplingStrategy::WithoutReplacement,
        &mut random_generator,
    );

    // split up again into batch/target arrays for return.
    let shuffled = ArrayView::from(&shuffled);
    let (batch, targets) = shuffled.split_at(Axis(1), batch_col_count);

    // done!
    (batch.into_owned(), targets.into_owned())
}

/// Function which runs a standard feed forward training process on a single
/// neural network with a given loss function for calculating error, as well as
/// a factory which can be used to define the optimisation strategy to use.
///
/// # Errors
/// Returns an `eidetic::Error` if the shapes of batches or targets don't agree with the network, or if the number of
/// rows in a batch doesn't match the number of rows in a targets tensor.
#[allow(clippy::too_many_arguments)]
pub fn train<N, O>(
    network: N,
    optimiser_factory: O,
    loss_function: &impl Loss,
    batch_train: Tensor<rank::Two>,
    targets_train: Tensor<rank::Two>,
    batch_test: &Tensor<rank::Two>,
    targets_test: &Tensor<rank::Two>,
    epochs: u16,
    eval_every: u16,
    batch_size: usize,
    seed: u64,
) -> Result<N>
where
    N: InitialisedOperation<Input = Tensor<rank::Two>, Output = Tensor<rank::Two>>
        + WithOptimiser<O>
        + Clone,
    <N as WithOptimiser<O>>::Trainable: TrainableOperation<Initialised = N>
        + Clone
        + for<'a> Forward<'a, Input = Tensor<rank::Two>, Output = Tensor<rank::Two>>,
    for<'a> <<N as WithOptimiser<O>>::Trainable as Forward<'a>>::Forward:
        ForwardOperation<Output = Tensor<rank::Two>>,
    for<'a> <<<N as WithOptimiser<O>>::Trainable as Forward<'a>>::Forward as ForwardOperation>::Backward:
        BackwardOperation,
{
    // check the input data is correctly shaped first (number of rows in the
    // batch should match number of rows in the targets).
    let (batch_train, targets_train) = (batch_train.0, targets_train.0);
    if (batch_train.nrows() != targets_train.nrows())
        || (batch_test.0.nrows() != targets_test.0.nrows())
    {
        return Err(Error(()));
    }

    // make the network trainable first.
    let mut best_loss: Option<ElementType> = None;
    let mut best_network: Option<N> = None;
    let mut network = network.with_optimiser(optimiser_factory);
    network.init(epochs);

    // loop number of epochs. For each one, permute data, generate batches
    // and every "eval_every" epochs, check against testing data.
    for e in 0..epochs {
        // potentially store the last model if this is an epoch where we may need to return to it.
        let last_model = if (e + 1) % eval_every == 0 {
            Some(network.clone().into_initialised())
        } else {
            None
        };

        // permute data first, using seed + epoch number for randomness.
        // then generate the batches, and for each one run a training pass for it.
        let (batch_train, targets_train) =
            permute_data(batch_train.clone(), &targets_train, seed + u64::from(e));
        for (batch, targets) in generate_batches(&batch_train, &targets_train, batch_size) {
            let (batch, targets) = (Tensor(batch), Tensor(targets));
            let (forward, output) = network.forward(batch)?;
            let (_, loss_gradient) = loss_function.loss(&output, &targets)?;
            let (backward, _) = forward.backward(loss_gradient)?;
            backward.optimise();
        }

        // if we're on an epoch that's evaluating the loss against the test batch,
        // then we will do this and early out if the loss worsens.
        if let Some(last_model) = last_model {
            // determine the loss against test data.
            let output = last_model.predict(batch_test.clone())?;
            let (loss, _) = loss_function.loss(&output, targets_test)?;

            // if we have a previous best loss and it's less than the
            // current loss, then early return previous network.
            if let Some(best_loss) = best_loss {
                if best_loss < loss.abs() {
                    return best_network.ok_or(Error(()));
                }
            }

            best_loss = Some(loss.abs());
            best_network = Some(last_model);
        }

        // Update the network to update the optimisers, etc. at the end of the epoch.
        if e < (epochs - 1) {
            network.end_epoch();
        }
    }

    // get the trained network out of the training wrapper.
    Ok(network.into_initialised())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_batches_with_size_greater_than_rows() {
        // Arrange
        let batch = Array::ones((3, 1));
        let targets = Array::ones((3, 1));

        // Act
        let mut iter = generate_batches(&batch, &targets, 4);

        // Assert
        let (batch, targets) = iter.next().unwrap();
        assert_eq!(batch.nrows(), 3);
        assert_eq!(targets.nrows(), 3);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_generate_batches_with_size_as_a_multiplier_of_rows() {
        // Arrange
        let batch = Array::ones((3, 1));
        let targets = Array::ones((3, 1));

        // Act
        let mut iter = generate_batches(&batch, &targets, 3);

        // Assert
        let (batch, targets) = iter.next().unwrap();
        assert_eq!(batch.nrows(), 3);
        assert_eq!(targets.nrows(), 3);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_generate_batches_with_size_as_a_non_multiplier_of_rows() {
        // Arrange
        let batch = Array::ones((3, 1));
        let targets = Array::ones((3, 1));

        // Act
        let mut iter = generate_batches(&batch, &targets, 2);

        // Assert
        let (batch, targets) = iter.next().unwrap();
        assert_eq!(batch.nrows(), 2);
        assert_eq!(targets.nrows(), 2);
        let (batch, targets) = iter.next().unwrap();
        assert_eq!(batch.nrows(), 1);
        assert_eq!(targets.nrows(), 1);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_permute_data() {
        // Arrange
        let batch = Array::from_iter((1_u16..=100).map(ElementType::from))
            .into_shape((100, 1))
            .unwrap();
        let targets = Array::from_iter((101_u16..=200).map(ElementType::from))
            .into_shape((100, 1))
            .unwrap();
        let seed = 42;

        // Act
        let (batch, targets) = permute_data(batch, &targets, seed);
        let expected = batch.mapv(|elem| 100.0 + elem);

        // Assert
        assert_eq!(targets, expected);
    }
}
