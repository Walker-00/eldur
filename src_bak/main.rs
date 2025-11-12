use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use thiserror::Error;

struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

#[derive(Error, Debug)]
enum TensorError {
    #[error("Incompatible Error: {0}")]
    IncompatibleShape(&'static str),
}

impl Tensor {
    pub fn new(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        let strides = Tensor::compute_strides(shape);
        Tensor {
            data: vec![0.0; size],
            shape: shape.to_vec(),
            strides,
        }
    }

    // Create a tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Self {
        Self::new(shape) // new() already creates zeros
    }

    // Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        let strides = Tensor::compute_strides(shape);
        Tensor {
            data: vec![1.0; size], // Fill with 1.0 instead of 0.0
            shape: shape.to_vec(),
            strides,
        }
    }

    // Create a tensor with specific value (like torch.full)
    pub fn full(shape: &[usize], value: f32) -> Self {
        let size = shape.iter().product();
        let strides = Tensor::compute_strides(shape);
        Tensor {
            data: vec![value; size],
            shape: shape.to_vec(),
            strides,
        }
    }

    // Create scalar tensor (0D) with value
    pub fn scalar(value: f32) -> Self {
        Tensor {
            data: vec![value],
            shape: vec![], // Empty shape for scalar (0D tensor)
            strides: vec![],
        }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    fn item(&self) -> Result<&f32, TensorError> {
        if self.data.len() == 1 {
            Ok(&self.data[0])
        } else {
            Err(TensorError::IncompatibleShape(
                "item() can only be called on tensors with exactly one element",
            ))
        }
    }

    fn item_owned(&self) -> Result<f32, TensorError> {
        if self.data.len() == 1 {
            Ok(self.data[0])
        } else {
            Err(TensorError::IncompatibleShape(
                "item() can only be called on tensors with exactly one element",
            ))
        }
    }
}

impl Tensor {
    // Create identity matrix (like torch.eye)
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor {
            data,
            shape: vec![n, n],
            strides: Tensor::compute_strides(&[n, n]),
        }
    }

    // Create tensor with values from range (like torch.arange)
    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        let mut data = Vec::new();
        let mut current = start;
        while current < end {
            data.push(current);
            current += step;
        }
        let shape = vec![data.len()];
        Tensor {
            data,
            shape: shape.clone(),
            strides: Tensor::compute_strides(&shape),
        }
    }

    // Create tensor with linear spacing (like torch.linspace)
    pub fn linspace(start: f32, end: f32, steps: usize) -> Self {
        let step_size = (end - start) / (steps - 1) as f32;
        let data: Vec<f32> = (0..steps).map(|i| start + i as f32 * step_size).collect();
        let shape = vec![steps];
        Tensor {
            data,
            shape: shape.clone(),
            strides: Tensor::compute_strides(&shape),
        }
    }
}

fn main() {
    // Like torch.zeros([2, 3])
    let zeros = Tensor::zeros(&[2, 3]);

    // Like torch.ones([4, 5])
    let ones = Tensor::ones(&[4, 5]);

    // Like torch.full([3, 3], 7.0)
    let full = Tensor::full(&[3, 3], 7.0);

    // Like torch.tensor(5.0)
    let scalar = Tensor::scalar(5.0);

    // Like torch.eye(3)
    let identity = Tensor::eye(3);

    // Like torch.arange(0, 10, 2)
    let range = Tensor::arange(0.0, 10.0, 2.0);

    // Like torch.linspace(0, 1, 5)
    let linear = Tensor::linspace(0.0, 1.0, 5);
}
