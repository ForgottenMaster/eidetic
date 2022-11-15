#[cfg(feature = "thiserror")]
use thiserror::Error;

/// The enumeration containing the possible errors that Eidetic can produce.
#[derive(Debug)]
#[cfg_attr(
    feature = "thiserror",
    derive(Error), 
    error("An error that can occur during runtime operation of Eidetic.")
)]
pub enum Error {}

/// A type alias which allows us to omit the error type when writing framework
/// function signatures in order to ensure everything is using Eidetic's Error
/// enumeration.
pub type Result<T> = core::result::Result<T, Error>;
