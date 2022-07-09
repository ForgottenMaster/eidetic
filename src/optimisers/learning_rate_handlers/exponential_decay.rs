use crate::private::Sealed;
use crate::ElementType;

/// A structure representing an exponentially decaying learning rate
/// which will decay per epoch from the given starting rate to the given
/// ending rate.
#[derive(Clone, Debug, PartialEq)]
pub struct LearningRateHandler {
    starting_rate: ElementType,
    ending_rate: ElementType,
    current_rate: ElementType,
    decay_per_epoch: ElementType,
}

impl LearningRateHandler {
    /// Constructs a new instance of an exponentially decaying learning rate.
    /// Takes the start and end rate to be lerped between over training.
    #[must_use]
    pub const fn new(starting_rate: ElementType, ending_rate: ElementType) -> Self {
        Self {
            starting_rate,
            ending_rate,
            current_rate: starting_rate,
            decay_per_epoch: 0.0,
        }
    }
}

impl Sealed for LearningRateHandler {}
impl super::LearningRateHandler for LearningRateHandler {
    fn learning_rate(&self) -> ElementType {
        self.current_rate
    }

    fn init(&mut self, epochs: u16) {
        self.decay_per_epoch =
            (self.ending_rate / self.starting_rate).powf(1.0 / (ElementType::from(epochs) - 1.0));
    }

    fn end_epoch(&mut self) {
        self.current_rate *= self.decay_per_epoch;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimisers::learning_rate_handlers::LearningRateHandler as BaseLearningRateHandler;

    #[test]
    fn test_initial_rate_is_starting_rate() {
        // Arrange
        let handler = LearningRateHandler::new(0.1, 0.05);

        // Assert
        assert_eq!(handler.learning_rate(), 0.1);
    }

    #[test]
    fn test_learning_rate_is_correct_after_first_epoch() {
        // Arrange
        let mut handler = LearningRateHandler::new(0.1, 0.05);

        // Act
        handler.init(10);
        handler.end_epoch();

        // Assert
        assert_eq!(handler.learning_rate(), 0.09258747122872905);
    }

    #[test]
    fn test_learning_rate_is_correct_after_half_epochs() {
        // Arrange
        let mut handler = LearningRateHandler::new(0.1, 0.05);

        // Act
        handler.init(10);
        (0..5).for_each(|_| handler.end_epoch());

        // Assert
        assert_eq!(handler.learning_rate(), 0.06803950000871886);
    }

    #[test]
    fn test_learning_rate_is_correct_after_all_epochs() {
        // Arrange
        let mut handler = LearningRateHandler::new(0.1, 0.05);

        // Act
        handler.init(10);
        (0..9).for_each(|_| handler.end_epoch());

        // Assert
        assert_eq!(handler.learning_rate(), 0.05000000000000002);
    }
}
