pub trait OptimiserFactory<T> {
    type Optimiser: Optimiser;
    fn instantiate(&self) -> Self::Optimiser;
}

pub trait Optimiser {
    type Parameter;
    fn optimise(&mut self, parameter: &mut Self::Parameter, gradient: &Self::Parameter);
}
