//! This module will contain the traits and structures for the various methods
//! of optimisation that can be used when updating an operation's parameter.

pub(crate) mod base;
pub(crate) mod null;

pub use null::OptimiserFactory as NullOptimiser;
