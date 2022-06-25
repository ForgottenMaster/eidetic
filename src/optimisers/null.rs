//! This module contains the optimiser factory/optimiser that does nothing.
//! This is a sensible default `OptimiserFactory` just so we can specify "something"
//! in the generic type of the initialised operation trait as the factory to use
//! in order to turn the initialised operation into a trainable one.

use crate::optimisers;
use crate::private::Sealed;
use core::marker::PhantomData;

#[derive(Debug, Default, PartialEq)]
pub struct OptimiserFactory(());

impl Sealed for OptimiserFactory {}
impl<T> optimisers::base::OptimiserFactory<T> for OptimiserFactory {}

/// This struct is the concrete optimiser that is produced by the
/// null `OptimiserFactory`.
pub struct Optimiser<T>(PhantomData<T>);

impl<T> Sealed for Optimiser<T> {}
impl<T> optimisers::base::Optimiser for Optimiser<T> {}
