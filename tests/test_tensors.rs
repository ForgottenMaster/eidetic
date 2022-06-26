use eidetic::tensors::*;
use eidetic::*;

#[test]
fn test_tensor_rank_0_construction() {
    let tensor = Tensor::<rank::Zero>::new(42.0);
    assert_eq!(tensor.into_iter().next().unwrap(), 42.0);
}

#[test]
fn test_tensor_rank_1_construction() {
    let tensor = Tensor::<rank::One>::new((1..=3u16).map(|elem| ElementType::from(elem)));
    let mut iter = tensor.into_iter();
    assert_eq!(iter.next().unwrap(), 1.0);
    assert_eq!(iter.next().unwrap(), 2.0);
    assert_eq!(iter.next().unwrap(), 3.0);
}

#[test]
fn test_tensor_rank_2_construction() {
    let tensor =
        Tensor::<rank::Two>::new((2, 2), (1..=4u16).map(|elem| ElementType::from(elem))).unwrap();
    let mut iter = tensor.into_iter();
    assert_eq!(iter.next().unwrap(), 1.0);
    assert_eq!(iter.next().unwrap(), 2.0);
    assert_eq!(iter.next().unwrap(), 3.0);
    assert_eq!(iter.next().unwrap(), 4.0);
}

#[test]
fn test_tensor_rank_3_construction() {
    let tensor =
        Tensor::<rank::Three>::new((2, 3, 2), (1..=12u16).map(|elem| ElementType::from(elem)))
            .unwrap();
    let expected = (1..=12u16)
        .map(|elem| ElementType::from(elem))
        .collect::<Vec<_>>();
    let output = tensor.into_iter().collect::<Vec<_>>();
    assert_eq!(expected, output);
}

#[test]
fn test_tensor_rank_4_construction() {
    let tensor = Tensor::<rank::Four>::new(
        (2, 3, 2, 2),
        (1..=24u16).map(|elem| ElementType::from(elem)),
    )
    .unwrap();
    let expected = (1..=24u16)
        .map(|elem| ElementType::from(elem))
        .collect::<Vec<_>>();
    let output = tensor.into_iter().collect::<Vec<_>>();
    assert_eq!(expected, output);
}

#[test]
fn test_tensor_rank_5_construction() {
    let tensor = Tensor::<rank::Five>::new(
        (2, 3, 2, 2, 3),
        (1..=72u16).map(|elem| ElementType::from(elem)),
    )
    .unwrap();
    let expected = (1..=72u16)
        .map(|elem| ElementType::from(elem))
        .collect::<Vec<_>>();
    let output = tensor.into_iter().collect::<Vec<_>>();
    assert_eq!(expected, output);
}

#[test]
fn test_tensor_rank_2_construction_failure() {
    assert!(
        Tensor::<rank::Two>::new((2, 2), (1..=5u16).map(|elem| ElementType::from(elem))).is_err()
    );
}

#[test]
fn test_tensor_rank_3_construction_failure() {
    assert!(
        Tensor::<rank::Three>::new((2, 3, 2), (1..=13u16).map(|elem| ElementType::from(elem)))
            .is_err()
    );
}

#[test]
fn test_tensor_rank_4_construction_failure() {
    assert!(Tensor::<rank::Four>::new(
        (2, 3, 2, 4),
        (1..=51u16).map(|elem| ElementType::from(elem))
    )
    .is_err());
}

#[test]
fn test_tensor_rank_5_construction_failure() {
    assert!(Tensor::<rank::Five>::new(
        (2, 3, 2, 4, 2),
        (1..=97u16).map(|elem| ElementType::from(elem))
    )
    .is_err());
}
