use eidetic::tensors::*;

#[test]
fn test_tensor_rank_0_construction() {
    let mut tensor = Tensor::<_, rank::Zero>::new(42);
    assert_eq!(tensor.iter().next().unwrap(), &42);
    assert_eq!(tensor.iter_mut().next().unwrap(), &mut 42);
    assert_eq!(tensor.into_iter().next().unwrap(), 42);
}

#[test]
fn test_tensor_rank_1_construction() {
    let mut tensor = Tensor::<_, rank::One>::new(1..=3);
    {
        let mut iter = tensor.iter();
        assert_eq!(iter.next().unwrap(), &1);
        assert_eq!(iter.next().unwrap(), &2);
        assert_eq!(iter.next().unwrap(), &3);
    }
    {
        let mut iter = tensor.iter_mut();
        assert_eq!(iter.next().unwrap(), &mut 1);
        assert_eq!(iter.next().unwrap(), &mut 2);
        assert_eq!(iter.next().unwrap(), &mut 3);
    }
    {
        let mut iter = tensor.into_iter();
        assert_eq!(iter.next().unwrap(), 1);
        assert_eq!(iter.next().unwrap(), 2);
        assert_eq!(iter.next().unwrap(), 3);
    }
}

#[test]
fn test_tensor_rank_2_construction() {
    let mut tensor = Tensor::<_, rank::Two>::new((2, 2), 1..=4).unwrap();
    {
        let mut iter = tensor.iter();
        assert_eq!(iter.next().unwrap(), &1);
        assert_eq!(iter.next().unwrap(), &2);
        assert_eq!(iter.next().unwrap(), &3);
        assert_eq!(iter.next().unwrap(), &4);
    }
    {
        let mut iter = tensor.iter_mut();
        assert_eq!(iter.next().unwrap(), &mut 1);
        assert_eq!(iter.next().unwrap(), &mut 2);
        assert_eq!(iter.next().unwrap(), &mut 3);
        assert_eq!(iter.next().unwrap(), &mut 4);
    }
    {
        let mut iter = tensor.into_iter();
        assert_eq!(iter.next().unwrap(), 1);
        assert_eq!(iter.next().unwrap(), 2);
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.next().unwrap(), 4);
    }
}

#[test]
fn test_tensor_rank_3_construction() {
    let tensor = Tensor::<_, rank::Three>::new((2, 3, 2), 1..=12).unwrap();
    let expected = (1..=12).collect::<Vec<_>>();
    let output = tensor.into_iter().collect::<Vec<_>>();
    assert_eq!(expected, output);
}

#[test]
fn test_tensor_rank_4_construction() {
    let tensor = Tensor::<_, rank::Four>::new((2, 3, 2, 2), 1..=24).unwrap();
    let expected = (1..=24).collect::<Vec<_>>();
    let output = tensor.into_iter().collect::<Vec<_>>();
    assert_eq!(expected, output);
}

#[test]
fn test_tensor_rank_5_construction() {
    let tensor = Tensor::<_, rank::Five>::new((2, 3, 2, 2, 3), 1..=72).unwrap();
    let expected = (1..=72).collect::<Vec<_>>();
    let output = tensor.into_iter().collect::<Vec<_>>();
    assert_eq!(expected, output);
}
