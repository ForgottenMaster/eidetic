use eidetic::activations::Linear;
use eidetic::operations::{
    BackwardOperation, ForwardOperation, InitialisedOperation, TrainableOperation,
    UninitialisedOperation, WithOptimiser,
};
use eidetic::optimisers::NullOptimiser;
use eidetic::tensors::{rank, Tensor};

#[test]
fn test_linear_operation_with_iter() {
    let linear = Linear::new();
    assert!(linear.with_iter([1.0, 4.0].into_iter()).is_ok());
}

#[test]
fn test_linear_operation_with_seed() {
    // test iter function
    let linear = Linear::new().with_seed_private(42, 2).0;
    assert_eq!(linear.iter().count(), 0);

    // test predictions
    let valid_tensor_1 =
        Tensor::<rank::Two>::new((2, 2), [2.0, 3.0, 4.0, 5.0].into_iter()).unwrap();
    let valid_tensor_2 =
        Tensor::<rank::Two>::new((3, 2), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0].into_iter()).unwrap();
    let invalid_tensor =
        Tensor::<rank::Two>::new((2, 3), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0].into_iter()).unwrap();
    assert_eq!(
        linear.predict(valid_tensor_1.clone()).unwrap(),
        valid_tensor_1.clone()
    );
    let prediction = linear.predict(valid_tensor_2.clone()).unwrap();
    assert_eq!(prediction, valid_tensor_2.clone());
    assert!(linear.predict(invalid_tensor.clone()).is_err());

    // map to trainable
    let mut linear = linear.with_optimiser(NullOptimiser::new());

    // test forward both with valid and invalid tensors
    assert_eq!(
        linear.forward(valid_tensor_1.clone()).unwrap().1,
        valid_tensor_1
    );
    assert_eq!(
        linear.forward(valid_tensor_2.clone()).unwrap().1,
        valid_tensor_2
    );
    assert!(linear.forward(invalid_tensor.clone()).is_err());

    // test into_initialised again
    let linear = linear.into_initialised();
    let mut input_iter = linear.iter();
    assert!(input_iter.next().is_none());

    // back to trainable to test the gradient is passed through correctly.
    let mut linear = linear.with_optimiser(NullOptimiser::new());
    assert_eq!(
        linear
            .forward(valid_tensor_1.clone())
            .unwrap()
            .0
            .backward(valid_tensor_1.clone())
            .unwrap()
            .1,
        valid_tensor_1
    );
    assert_eq!(
        linear
            .forward(valid_tensor_2.clone())
            .unwrap()
            .0
            .backward(valid_tensor_2.clone())
            .unwrap()
            .1,
        valid_tensor_2
    );
    assert!(linear
        .forward(valid_tensor_1.clone())
        .unwrap()
        .0
        .backward(invalid_tensor)
        .is_err());

    // finally testing the optimise call after the backward pass (which does nothing).
    linear
        .forward(valid_tensor_1.clone())
        .unwrap()
        .0
        .backward(valid_tensor_1)
        .unwrap()
        .0
        .optimise();
}
