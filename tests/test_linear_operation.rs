use eidetic::operations::{Linear, UninitialisedOperation};

#[test]
fn test_linear_operation_in_isolation_with_iter() {
    let operation = Linear::new(42);
    operation.with_iter([1.0].into_iter()).unwrap();
}

#[test]
fn test_linear_operation_in_isolation_with_seed() {
    let operation = Linear::<f64>::new(42);
    operation.with_seed(42).unwrap();
}
