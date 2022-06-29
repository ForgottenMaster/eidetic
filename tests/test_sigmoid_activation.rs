use eidetic::activations::Sigmoid;
use eidetic::operations::{
    BackwardOperation, ForwardOperation, InitialisedOperation, TrainableOperation,
    UninitialisedOperation, WithOptimiser,
};
use eidetic::optimisers::NullOptimiser;
use eidetic::tensors::{rank, Tensor};

#[test]
fn test_sigmoid_operation_with_iter() {
    let sigmoid = Sigmoid::new();
    assert!(sigmoid.with_iter([1.0, 4.0].into_iter()).is_ok());
}

#[test]
fn test_sigmoid_operation_with_seed() {
    // test iter function
    let sigmoid = Sigmoid::new().with_seed_private(42, 3).0;
    assert_eq!(sigmoid.iter().count(), 0);

    // test predictions
    let valid_tensor = Tensor::<rank::Two>::new((1, 3), [-6.0, 0.0, 6.0].into_iter()).unwrap();
    let invalid_tensor =
        Tensor::<rank::Two>::new((1, 4), [-6.0, 0.0, 6.0, 7.0].into_iter()).unwrap();

    let valid_tensor_output = Tensor::<rank::Two>::new(
        (1, 3),
        #[cfg(feature = "f32")]
        [0.002472623, 0.5, 0.9975274].into_iter(),
        #[cfg(not(feature = "f32"))]
        [0.0024726231566347743, 0.5, 0.9975273768433653].into_iter(),
    )
    .unwrap();

    assert_eq!(
        sigmoid.predict(valid_tensor.clone()).unwrap(),
        valid_tensor_output.clone()
    );
    assert!(sigmoid.predict(invalid_tensor.clone()).is_err());

    // map to trainable
    let mut sigmoid = sigmoid.with_optimiser(NullOptimiser::new());

    // test forward both with valid and invalid tensors
    assert_eq!(
        sigmoid.forward(valid_tensor.clone()).unwrap().1,
        valid_tensor_output
    );
    assert!(sigmoid.forward(invalid_tensor.clone()).is_err());

    // test into_initialised again
    let sigmoid = sigmoid.into_initialised();
    let mut input_iter = sigmoid.iter();
    assert!(input_iter.next().is_none());

    // back to trainable to test the gradient is passed through correctly.
    let output_gradient = Tensor::<rank::Two>::new((1, 3), [1.0, 1.0, 1.0].into_iter()).unwrap();
    let expected_input_gradient = Tensor::<rank::Two>::new(
        (1, 3),
        #[cfg(feature = "f32")]
        [0.0024665091, 0.25, 0.0024664658].into_iter(),
        #[cfg(not(feature = "f32"))]
        [0.002466509291360048, 0.25, 0.002466509291359931].into_iter(),
    )
    .unwrap();
    let mut sigmoid = sigmoid.with_optimiser(NullOptimiser::new());
    assert_eq!(
        sigmoid
            .forward(valid_tensor.clone())
            .unwrap()
            .0
            .backward(output_gradient.clone())
            .unwrap()
            .1,
        expected_input_gradient
    );
    assert!(sigmoid
        .forward(valid_tensor.clone())
        .unwrap()
        .0
        .backward(invalid_tensor)
        .is_err());

    // finally testing the optimise call after the backward pass (which does nothing).
    sigmoid
        .forward(valid_tensor.clone())
        .unwrap()
        .0
        .backward(valid_tensor)
        .unwrap()
        .0
        .optimise();
}
