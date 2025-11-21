pub trait TensorOps {
    type Tensor: TensorOps;

    fn add(&self, other: Self::Tensor) -> Self::Tensor;

    fn sum(&self) -> f32;

    fn mean(&self) -> f32;

    fn swapaxes(&self, axis1: usize, axis2: usize) -> Self::Tensor;

    fn narrow(&self, axis: usize, start: usize, end: usize) -> Self::Tensor;
}

pub trait ParTensorOps {
    type Tensor: TensorOps;

    fn par_add(&self, other: Self::Tensor) -> Self::Tensor;

    fn par_sum(&self) -> f32;

    // fn mean(&self) -> f32;
    //
    // fn swapaxes(&self, axis1: usize, axis2: usize) -> Self::Tensor;
    //
    // fn narrow(&self, axis: usize, start: usize, end: usize) -> Self::Tensor;
}
