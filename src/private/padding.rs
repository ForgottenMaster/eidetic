use crate::ElementType;
use ndarray::{Array, Axis, Ix1};

pub fn _pad_1d(input: &Array<ElementType, Ix1>, num: usize) -> Array<ElementType, Ix1> {
    let padding = Array::from_iter(core::iter::repeat(0.0).take(num));
    let mut output = padding.clone();
    output.append(Axis(0), input.view()).unwrap();
    output.append(Axis(0), padding.view()).unwrap();
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_1d() {
        // Arrange
        let input = Array::from_iter([1.0, 2.0, 3.0, 4.0, 5.0]);
        let expected = Array::from_iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0]);

        // Act
        let output = _pad_1d(&input, 1);

        // Assert
        assert_eq!(output, expected);
    }
}
