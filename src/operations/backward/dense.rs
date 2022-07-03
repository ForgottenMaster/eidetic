pub struct Operation<T, U, V> {
    pub(crate) _weight_multiply: T,
    pub(crate) _bias_add: U,
    pub(crate) _activation_function: V,
}
