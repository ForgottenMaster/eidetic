pub trait OptimiserFactory<T> {
    type Optimiser: Optimiser<T>;
    fn instantiate(&self) -> Self::Optimiser;
}

pub trait Optimiser<T> {
    fn optimise(&mut self, parameter: &mut T, gradient: &T);
}
