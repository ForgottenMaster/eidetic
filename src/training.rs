//! This module contains training related functionality for
//! taking an initialised network and training it over a certain
//! number of epochs with a certain optimisation strategy, etc.

use crate::ElementType;
use ndarray::{Array, ArrayView, Axis, Ix2};
use ndarray_rand::{RandomExt, SamplingStrategy};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn _generate_batches<'a>(
    batch: &'a Array<ElementType, Ix2>,
    targets: &'a Array<ElementType, Ix2>,
    size: usize,
) -> impl Iterator<Item = (Array<ElementType, Ix2>, Array<ElementType, Ix2>)> + 'a {
    batch
        .axis_chunks_iter(Axis(0), size)
        .zip(targets.axis_chunks_iter(Axis(0), size))
        .map(|(view1, view2)| (view1.to_owned(), view2.to_owned()))
}

fn _permute_data(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_batches_with_size_greater_than_rows() {
        // Arrange
        let batch = Array::ones((3, 1));
        let targets = Array::ones((3, 1));

        // Act
        let mut iter = _generate_batches(&batch, &targets, 4);

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
        let mut iter = _generate_batches(&batch, &targets, 3);

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
        let mut iter = _generate_batches(&batch, &targets, 2);

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
        let (batch, targets) = _permute_data(batch, &targets, seed);
        let expected = batch.mapv(|elem| 100.0 + elem);

        // Assert
        assert_eq!(targets, expected);
    }
}
