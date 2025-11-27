pub trait TensorOps {
    type SeqTensor;

    fn add(&self, other: &Self::SeqTensor) -> Self::SeqTensor;

    fn sum(&self) -> f32;

    fn mean(&self) -> f32;

    fn swapaxes(&self, axis1: usize, axis2: usize) -> Self::SeqTensor;

    fn narrow(&self, axis: usize, start: usize, end: usize) -> Self::SeqTensor;
}

pub trait ParTensorOps {
    type ParTensor: TensorOps;

    fn par_add(&self, other: &Self::ParTensor) -> Self::ParTensor;

    fn par_sum(&self) -> f32;

    // fn mean(&self) -> f32;
    //
    // fn swapaxes(&self, axis1: usize, axis2: usize) -> Self::Tensor;
    //
    // fn narrow(&self, axis: usize, start: usize, end: usize) -> Self::Tensor;
}
