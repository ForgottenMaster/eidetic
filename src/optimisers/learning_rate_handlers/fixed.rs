use crate::private::Sealed;
use crate::ElementType;

/// This is a provider for a constant learning rate that doesn't change or degrade
/// based on epoch. It's the most basic type of learning rate handling.
#[derive(Debug, PartialEq)]
pub struct LearningRateHandler {
    learning_rate: ElementType,
}

impl LearningRateHandler {
    /// Constructs a new instance of the `LearningRateHandler` with a fixed
    /// learning rate that it always reports when asked for.
    #[must_use]
    pub const fn new(learning_rate: ElementType) -> Self {
        Self { learning_rate }
    }
}

impl Sealed for LearningRateHandler {}
impl super::LearningRateHandler for LearningRateHandler {
    fn learning_rate(&self) -> ElementType {
        self.learning_rate
    }

    fn init(&mut self, _epochs: u32) {}

    fn end_epoch(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimisers::learning_rate_handlers::LearningRateHandler as BaseLearningRateHandler;

    #[test]
    fn test_init() {
        // Arrange
        let mut fixed = LearningRateHandler::new(0.1);

        // Act
        fixed.init(0);

        // Assert
        assert_eq!(fixed.learning_rate(), 0.1);
    }

    #[test]
    fn test_end_epoch() {
        // Arrange
        let mut fixed = LearningRateHandler::new(0.1);
        let expected = LearningRateHandler::new(0.1);

        // Act
        fixed.end_epoch();

        // Assert
        assert_eq!(fixed.learning_rate(), 0.1);
        assert_eq!(fixed, expected);
    }
}
