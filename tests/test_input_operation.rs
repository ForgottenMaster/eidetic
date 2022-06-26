use eidetic::operations::{Input, UninitialisedOperation};

#[test]
fn test_input_operation() {
    let input = Input::new(42);
    let (_input, neurons) = input.with_iter([1.0, 4.0].into_iter()).unwrap();
    assert_eq!(neurons, 42);
}
