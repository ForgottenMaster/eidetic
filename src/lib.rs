#![cfg_attr(not(feature = "std"), no_std)]
#![deny(warnings, missing_docs, clippy::pedantic, clippy::nursery)]

//! Eidetic is a simple pure rust library for neural networks and deep learning.
//! Implemented alongside working through the "Deep learning from scratch in Python" book in order to get hands on with it.
//! This API is implemented with the following pillars:
//! 1. **Stability** - No dependency is included in the public API unless it has reached 1.0. This ensures that the public API is only tied to our versioning.
//! 2. **Embeddable** - The library doesn't use the standard library or a runtime so can be used in embedded environments (std feature can be turned on when required though).
//! 3. **Foolproof** - We will make copious use of typestates and error handling to ensure that the API cannot be misused in any way. Where possible, correct API usage will be verified by the compiler. Otherwise it will be verified by returning Results at runtime.
//! 4. **Correctness** - We make use of unit testing and documentation testing to verify that the API is correct and functions as expected. Any example code in documentation will be correct and compile.

pub mod activations;
pub mod layers;
pub mod loss;
pub mod operations;
pub mod optimisers;
mod private;
pub mod tensors;
pub mod training;

#[cfg(feature = "thiserror")]
use thiserror::Error;

/// This is a generic error that's emitted for when something in Eidetic
/// goes wrong. Having a custom error type means we can typedef Result too.
#[derive(Debug, Eq, PartialEq)]
#[cfg_attr(feature = "thiserror", derive(Error))]
#[cfg_attr(feature = "thiserror", error("An error that can occur during runtime operation of Eidetic. Since the API uses typestates and catches issues at compile time, this will usually be an invalid shape (e.g. incorrect column count in the data)."))]
pub struct Error(pub(crate) ());

/// This is the Result type alias defined by Eidetic to
/// hard code the error type to be `eidetic::Error`.
pub type Result<T> = core::result::Result<T, Error>;

/// This is defined to be the element type to be used by the library.
/// This is the default type to be used.
#[cfg(not(feature = "f32"))]
pub type ElementType = f64;

/// This is defined to be the element type to be used by the library.
/// This is used as the type only when the "f32" feature is enabled to save memory.
#[cfg(feature = "f32")]
pub type ElementType = f32;
