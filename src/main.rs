use ndarray::{ArrayD, IxDyn};
use num_traits::{One, Zero};

pub struct Tensor<T> {
    pub data: ArrayD<T>,
}

impl<T: Clone + Zero + One> Tensor<T> {
    pub fn from_array(data: ArrayD<T>) -> Self {
        Self { data }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::zeros(IxDyn(shape)),
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::ones(IxDyn(shape)),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn get(&self, index: &[usize]) -> Option<&T> {
        self.data.get(index)
    }
}

fn main() {
    println!("Hello, world!");
}
