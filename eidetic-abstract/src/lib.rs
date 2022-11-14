//! This crate contains the abstract functionality for Eidetic which is independent of any specific
//! concrete Backend. This defines the way that networks can be built and trained, but delegates the
//! actual tensor storage and low-level operations used by layers to the backend currently in use.
//!
//! ```
//! todo!("Sample of building a network");
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(warnings, missing_docs, clippy::all)]

mod backend;
mod backend_data_type;
mod tensor;

pub use backend::*;
pub use backend_data_type::*;
pub use tensor::*;
