use eidetic::operations::{Input, UninitialisedOperation};

#[test]
fn test_input_operation_with_iter() {
    let input = Input::new(2);
    assert!(input.with_iter([1.0, 4.0].into_iter()).is_ok());
}

#[test]
fn test_input_operation_with_seed() {
    let input = Input::new(2);
    assert_eq!(input.with_seed(42), ());
}
