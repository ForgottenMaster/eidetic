#![cfg_attr(not(feature = "std"), no_std)]
#![deny(warnings, missing_docs)]

//! Eidetic is a simple pure rust library for neural networks and deep learning.
//! Implemented alongside working through the "Deep learning from scratch in Python" book in order to get hands on with it.
//! This API is implemented with the following pillars:
//! 1. **Stability** - No dependency is included in the public API unless it has reached 1.0. This ensures that users have a stable public API to work with
//! 2. **Embeddable** - The library doesn't use the standard library or a runtime so can be used in embedded environments
//! 3. **Foolproof** - We will make copious use of typestates and error handling to ensure that the API cannot be misused in any way. Where possible correct API usage will be verified by the compiler
//! 4. **Correctness** - We make use of unit testing and documentation testing to verify that the API is correct and functions as expected. Any example code in documentation will be correct and compile
//! 5. **Well Documented** - Every function, type, module should be documented and where possible include code example/documentation tests.

mod errors;
mod tensor;

// re-exports
pub use errors::TensorConstructionError;
pub use tensor::Tensor2;
