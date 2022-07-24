#![cfg_attr(not(feature = "std"), no_std)]
#![deny(warnings, missing_docs, clippy::pedantic, clippy::nursery)]

//! Eidetic is a simple pure rust library for neural networks and deep learning.
//! Implemented alongside working through the "Deep learning from scratch in Python" book in order to get hands on with it.
//! This API is implemented with the following pillars:
//! 1. **Stability** - No dependency is included in the public API unless it has reached 1.0. This ensures that the public API is only tied to our versioning.
//! 2. **Embeddable** - The library doesn't use the standard library or a runtime so can be used in embedded environments (std feature can be turned on when required though).
//! 3. **Foolproof** - We will make copious use of typestates and error handling to ensure that the API cannot be misused in any way. Where possible, correct API usage will be verified by the compiler. Otherwise it will be verified by returning Results at runtime.
//! 4. **Correctness** - We make use of unit testing and documentation testing to verify that the API is correct and functions as expected. Any example code in documentation will be correct and compile.
//!
//! # Concepts
//! The following sections are the main concepts of the Eidetic library which will be useful as a starting point to use it. The API is designed to only be used in one way (the correct way) and thus it shouldn't
//! be necessary to explain every type here, but just the starting points.
//!
//! ### Tensors
//! In deep learning, data is represented in tensors which have a rank encoded into them. The rank of a tensor is how many dimensions it has. A rank 0 tensor is a single value and is known as a scalar. A rank 1
//! tensor consists of 1 dimension (length) which can represent a vector. A rank 2 tensor has width and height (rows and columns) and such can represent a matrix, etc. Tensors are all represented by the `Tensor` type
//! which is generic over the rank.
//!
//! The API takes the input in as a `Tensor` and produces the output as a `Tensor` also. This is so we can hide the internal implementation details of how the data is actually stored behind the scenes. The construction
//! of a Tensor takes a shape (for rank >= 2) and an iterator of values and will attempt to construct the Tensor. For rank 0 tensors we take a single value/scalar instead of an iterator and it is infallible. For rank 1 tensors we
//! are taking the entire iterator of elements as the length, and this too is infallible.
//!
//! For rank 2 or greater tensors however, these can be fallible because the shape requested might not match the number of elements inside the given iterator. Therefore for rank 2 or greater, the new function on
//! Tensor will return a result that must be handled.
//!
//! Examples of constructing some tensors are as follows. Ranks >2 follow the same pattern as Rank 2 tensors by taking their shape as a tuple of the appropriate arity.
//!
//! ```
//! use eidetic::tensors::{Tensor, rank};
//! let rank_0 = Tensor::<rank::Zero>::new(1.0);
//! let rank_1 = Tensor::<rank::One>::new([1.0, 2.0, 3.0]);
//! let rank_2 = Tensor::<rank::Two>::new((2, 2), [1.0, 2.0, 3.0, 4.0]).unwrap();
//! let rank_3 = Tensor::<rank::Three>::new((2, 2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
//! ```
//!
//! For getting data out of Eidetic, a Tensor supports the `IntoIterator` trait so can be used inside a for loop if needed as follows:
//!
//! ```
//! use eidetic::ElementType;
//! use eidetic::tensors::{Tensor, rank};
//! let rank_2 = Tensor::<rank::Two>::new((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
//! let tensor_sum: ElementType = rank_2.into_iter().sum();
//! assert_eq!(tensor_sum, 21.0);
//! ```
//!
//! Unfortunately at this time, the API for processing the data produced by Eidetic is limited to a flattened iterator and doesn't support more complex
//! interactions like iterating rows/columns/etc. This could be handled by your linear algebra library of choice if needed.
//!
//! ### `ElementType`
//! As you may have noticed in the above example, the data type that Eidetic works with is type-aliased to `eidetic::ElementType` which defaults to f64.
//! You can opt to use f32 instead however, by enabling the Cargo feature *f32* which will use the smaller but less accurate data type for devices where memory is constrained.
//! Eidetic doesn't work with any custom data types to avoid annoying trait bounds, etc. we would need for the generics so any data conversion must be done before providing the data to
//! Eidetic, or after getting the data from Eidetic.
//!
//! ### Operation Chain
//! In Eidetic, there's no dedicated "Network" type and instead the API operates on a chain of operations. This functionality is provided by the `Chain` trait and an operation chain
//! *MUST* begin with the `Input` layer due to the unique semantics it provides. An example of a Linear regression type of network can then be built as follows:
//!
//! ```
//! use eidetic::activations::{Linear};
//! use eidetic::layers::{Chain, Dense, Input};
//! let network = Input::new(2) // 2 input columns/features/neurons
//!        .chain(Dense::new(1, Linear::new())); // 1 output column/feature, using simple linear activation function
//! ```
//!
//! Once a chain of these layers is built, the "network" is in an uninitialised state, and can be initialised via one of the initialisation functions on the `UninitialisedOperation` trait
//! which allows either direct initialisation from pre-trained weights, or generation of weights using a random seed (uses Xavier initialisation).
//!
//! ### Typestates
//! The Eidetic API makes use of typestates to ensure that a network can't be trained unless it's initialised with weights first, which it can only do once the structure of the network is fixed (an operation
//! chain can't be modified once initialised).
//!
//! More information can be found in the appropriate documentation of the traits (`UninitialisedOperation`, `InitialisedOperation`, etc.), or by reading the examples in the examples directory of the library, but the following
//! shows a working example of the type states at work.
//!
//! ```
//! use eidetic::activations::{Linear};
//! use eidetic::layers::{Chain, Dense, Input};
//! use eidetic::operations::{BackwardOperation, Forward, ForwardOperation, TrainableOperation, UninitialisedOperation, WithOptimiser};
//! use eidetic::optimisers::NullOptimiser;
//! # use eidetic::tensors::{Tensor, rank};
//! let network = Input::new(2).chain(Dense::new(1, Linear::new())); // Uninitialised linear regression network with 2 input neurons and 1 output neuron.
//! let network = network.with_seed(42); // Initialise the weights of the network with a random seed (can use with_iter if weights are already known and stored to a file for example).
//! let mut network = network.with_optimiser(NullOptimiser::new()); // Providing an optimiser allows the network to be trained. Null optimiser does nothing, but others are for example SGD, SGDMomentum, etc.
//! # let input_data = Tensor::<rank::Two>::new((3, 2), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
//! let (forward_pass, _) = network.forward(input_data).unwrap(); // Forward pass taking the input data and producing output data. Returns a result since the amount of elements in the provided Tensor may not match expected shape at runtime.
//! # let output_gradient = Tensor::<rank::Two>::new((3, 1), [1.0, 1.0, 1.0]).unwrap();
//! let (backward_pass, _) = forward_pass.backward(output_gradient).unwrap(); // Takes the initial output gradient for the backward pass and runs it back through the network.
//! backward_pass.optimise(); // Apply optimisation using the gradients that were calculated.
//! let network = network.into_initialised(); // Once we're done training, we can put it back into an "initialised" typestate which will allow us to get access to the weights in the network.
//! ```
//!
//! # Examples
//! All the examples can be found inside the "examples" directory and run through the standard procedure of:
//!
//! ```ignore
//! cargo run --release --example <example_name> --features="features-required-by-example"
//! ```
//!
//! Note that if you try to run an example that requires a feature to be active (for example to download an additional crate) then it will tell you about it.

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
