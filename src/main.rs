use rayon::prelude::*;
use simdeez::prelude::*;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.par_iter().product();
        let strides = Self::compute_strides(shape);
        Self {
            data: vec![0.0; size],
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        assert_eq!(data.len(), shape.par_iter().product());
        let strides = Self::compute_strides(shape);

        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];

        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        strides
    }

    pub fn index(&self, idx: &[usize]) -> usize {
        assert_eq!(idx.len(), self.shape.len());
        // idx.par_iter().zip(self.strides.iter()).
        idx.iter()
            .zip(self.strides.iter())
            .map(|(i, s)| i * s)
            .sum()
    }

    pub fn get(&self, idx: &[usize]) -> f32 {
        self.data[self.index(idx)]
    }

    pub fn set(&mut self, idx: &[usize], value: f32) {
        let i = self.index(idx);
        self.data[i] = value;
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        simd_runtime_generate!(
            assert_eq!(self.data.len(), other.data.len());
        );
        todo!()
    }
}

fn main() {}
