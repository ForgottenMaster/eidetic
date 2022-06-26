use eidetic::operations::{
    InitialisedOperation, Input, TrainableOperation, UninitialisedOperation,
};
use eidetic::optimisers::NullOptimiser;
use eidetic::tensors::{rank, Tensor};

#[test]
fn test_input_operation_with_iter() {
    let input = Input::new(2);
    assert!(input.with_iter([1.0, 4.0].into_iter()).is_ok());
}

#[test]
fn test_input_operation_with_seed() {
    // test iter function
    let input = Input::new(2).with_seed(42);
    assert_eq!(input.iter().count(), 0);

    // test predictions
    let valid_tensor_1 =
        Tensor::<rank::Two>::new((2, 2), [2.0, 3.0, 4.0, 5.0].into_iter()).unwrap();
    let valid_tensor_2 =
        Tensor::<rank::Two>::new((3, 2), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0].into_iter()).unwrap();
    let invalid_tensor =
        Tensor::<rank::Two>::new((2, 3), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0].into_iter()).unwrap();
    assert_eq!(
        input.predict(valid_tensor_1.clone()).unwrap(),
        valid_tensor_1.clone()
    );
    let prediction = input.predict(valid_tensor_2.clone()).unwrap();
    assert_eq!(prediction, valid_tensor_2.clone());
    assert!(input.predict(invalid_tensor.clone()).is_err());

    // map to trainable
    let mut input = input.with_optimiser(NullOptimiser::new());

    // test forward both with valid and invalid tensors
    assert_eq!(
        input.forward(valid_tensor_1.clone()).unwrap().1,
        valid_tensor_1
    );
    assert_eq!(
        input.forward(valid_tensor_2.clone()).unwrap().1,
        valid_tensor_2
    );
    assert!(input.forward(invalid_tensor.clone()).is_err());

    // test into_initialised again
    let input = input.into_initialised();
    let valid_tensor_3 =
        Tensor::<rank::Two>::new((3, 2), [17.0, 18.0, 19.0, 20.0, 21.0, 22.0]).unwrap();
    assert_eq!(
        input.predict(valid_tensor_3.clone()).unwrap(),
        valid_tensor_3.clone()
    );
}
