pub trait SeqTensorOps {
    type SeqTensor: SeqTensorOps;

    fn add(&self, other: Self::SeqTensor) -> Self::SeqTensor;

    fn sum(&self) -> f32;

    fn mean(&self) -> f32;

    fn swapaxes(&self, axis1: usize, axis2: usize) -> Self::SeqTensor;

    fn narrow(&self, axis: usize, start: usize, end: usize) -> Self::SeqTensor;
}
