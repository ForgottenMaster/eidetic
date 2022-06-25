use crate::operations::uninitialised;
use crate::private::Sealed;
use crate::void::Void;
use core::marker::PhantomData;

pub struct Linear<T>(PhantomData<T>);

impl<T> Sealed for Linear<T> {}

impl<T> uninitialised::Operation for Linear<T> {
    type Element = T;
    type Error = Void;
    type Initialised = ();

    fn output_neuron_count(&self) -> usize {
        0
    }
}
