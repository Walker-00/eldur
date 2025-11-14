use std::iter;

use pulp::{Arch, WithSimd};
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
        assert_eq!(self.data.len(), other.data.len());
        let mut result = vec![0.0f32; self.data.len()];

        struct Impl<'a> {
            out: &'a mut [f32],
            a: &'a [f32],
            b: &'a [f32],
        }

        impl WithSimd for Impl<'_> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self { out, a, b } = self;

                let (out0, out1) = S::as_mut_simd_f32s(out);
                let (a0, a1) = S::as_simd_f32s(a);
                let (b0, b1) = S::as_simd_f32s(b);

                for (out, (a, b)) in iter::zip(out0, iter::zip(a0, b0)) {
                    *out = simd.add_f32s(*a, *b);
                }

                for (out, (a, b)) in iter::zip(out1, iter::zip(a1, b1)) {
                    *out = *a + *b;
                }
            }
        }

        Arch::new().dispatch(Impl {
            out: &mut result,
            a: &self.data,
            b: &other.data,
        });

        Tensor::from_vec(result, &self.shape)
    }

    pub fn sum(&self) -> f32 {
        struct Impl<'a> {
            input: &'a [f32],
        }

        impl WithSimd for Impl<'_> {
            type Output = f32;

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Self { input } = self;

                let (input0, input1) = S::as_simd_f32s(input);
                let (input04, input01) = pulp::as_arrays::<4, _>(input0);

                let mut sum0 = simd.splat_f32s(0.0);
                let mut sum1 = simd.splat_f32s(0.0);
                let mut sum2 = simd.splat_f32s(0.0);
                let mut sum3 = simd.splat_f32s(0.0);

                for [input0, input1, input2, input3] in input04 {
                    sum0 = simd.add_f32s(sum0, *input0);
                    sum1 = simd.add_f32s(sum1, *input1);
                    sum2 = simd.add_f32s(sum2, *input2);
                    sum3 = simd.add_f32s(sum3, *input3);
                }

                sum0 = simd.add_f32s(sum0, sum1);
                sum2 = simd.add_f32s(sum2, sum3);

                sum0 = simd.add_f32s(sum0, sum2);
                todo!()
            }
        }
        todo!()
    }
}

fn main() {}
