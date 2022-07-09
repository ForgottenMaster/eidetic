//! This module will contain the traits and structures for the various methods
//! of optimisation that can be used when updating an operation's parameter.

pub(crate) mod base;
pub mod learning_rate_handlers;
pub(crate) mod null;
mod sgd;
mod sgd_momentum;

pub use null::OptimiserFactory as NullOptimiser;
pub use sgd::OptimiserFactory as SGD;
pub use sgd_momentum::OptimiserFactory as SGDMomentum;
