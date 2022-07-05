use crate::operations::{forward, BackwardOperation};
use crate::private::Sealed;

#[derive(Debug, PartialEq)]
pub struct Operation<'a> {
    pub(crate) _forward: forward::dropout::Operation<'a>,
}

impl<'a> Sealed for Operation<'a> {}

impl<'a> BackwardOperation for Operation<'a> {
    fn optimise(self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::{initialised, trainable};
    use crate::tensors::{rank, Tensor};

    #[test]
    fn test_optimise() {
        // Arrange
        let mut backing = trainable::dropout::Operation {
            initialised: initialised::dropout::Operation {
                keep_probability: 0.8,
                seed: None,
            },
        };
        let backward = Operation {
            _forward: forward::dropout::Operation {
                _borrow: &mut backing,
                mask: Tensor::<rank::Two>::new((1, 3), [1.0, 2.0, 3.0]).unwrap(),
            },
        };

        // Act
        backward.optimise();
    }
}
