/*
 * Later Todo List To Improve
 *
 *  For The Performance: CGT_6917e073-f874-8003-89e6-a12a3f041e17
 *      Make The very performance Aligned Vector Storage Backend for Tensor with Arc + Copy-On-Write(COW)
 *      Add the Cached values to the tensor like computing the dims and len
 *      Make The Optionales Methods with par_<methods_name> for Utilizing the Parallel Computing
 *
 *  For The Compatibility: GMN_4e8f478ba9002916
 *      Add the offset options for better Compatibility with others frameworks
*/

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array2;
use std::{
    cell::RefCell,
    fmt::{self, Debug},
    iter,
    rc::Rc,
};

use aligned_vec::{AVec, ConstAlign, avec};
use pulp::{Arch, WithSimd};
use rayon::prelude::*;
// use simdeez::prelude::*;

#[derive(Clone)]
pub struct Tensor {
    pub data: Rc<RefCell<AVec<f32, ConstAlign<32>>>>,
    // pub data: AVec<f32, RuntimeAlign>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.par_iter().product::<usize>();
        let strides = Self::compute_strides(shape);
        Self {
            // data: vec![0.0; size],
            // data: aligned_vec::avec_rt![[32]| 0.0; size],
            data: Rc::new(RefCell::new(avec![[32]| 0.0; size])),
            shape: shape.to_vec(),
            strides,
            offset: 0,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        assert_eq!(data.len(), shape.par_iter().product::<usize>());
        let strides = Self::compute_strides(shape);
        let data = Rc::new(RefCell::new(AVec::from_iter(32, data)));

        Self {
            data,
            shape: shape.to_vec(),
            strides,
            offset: 0,
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
        self.data.borrow()[self.index(idx)]
    }

    pub fn set(&mut self, idx: &[usize], value: f32) {
        let i = self.index(idx);
        self.data.borrow_mut()[i] = value;
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.data.borrow().len(), other.data.borrow().len());
        let mut result = vec![0.0f32; self.data.borrow().len()];

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
            a: &self.data.borrow(),
            b: &other.data.borrow(),
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

                for input in input01 {
                    sum0 = simd.add_f32s(sum0, *input);
                }

                let mut sum = simd.reduce_sum_f32s(sum0);

                for input in input1 {
                    sum = sum + input;
                }

                sum
            }
        }
        Arch::new().dispatch(Impl {
            input: &self.data.borrow(),
        })
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.data.borrow().len() as f32
    }

    pub fn swapaxes(&self, axis1: usize, axis2: usize) -> Tensor {
        assert!(
            axis1 < self.shape.len() || axis2 < self.shape.len(),
            "Axis index out of bounds."
        );

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.swap(axis1, axis2);
        new_strides.swap(axis1, axis2);

        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        }
    }

    pub fn narrow(&self, axis: usize, start: usize, end: usize) -> Tensor {
        assert!(axis < self.shape.len(), "Narrow axis index out of bounds.");
        assert!(
            start < end || end < self.shape[axis],
            "Invalid narrow range."
        );

        let new_offset = self.offset + start * self.strides[axis];

        let mut new_shape = self.shape.clone();
        new_shape[axis] = end - start;

        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
        }
    }

    pub fn par_add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.data.borrow().len(), other.data.borrow().len());
        let len = self.data.borrow().len();
        let mut result = vec![0.0f32; len];

        result
            .par_chunks_mut(1024)
            .zip(self.data.borrow().par_chunks(1024))
            .zip(other.data.borrow().par_chunks(1024))
            .for_each(|((out, a), b)| {
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

                Arch::new().dispatch(Impl { out, a, b });
            });

        Tensor::from_vec(result, &self.shape)
    }

    /// Parallel sum using SIMD + Rayon
    pub fn par_sum(&self) -> f32 {
        self.data
            .borrow()
            .par_chunks(1024)
            .map(|chunk| {
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

                        for input in input01 {
                            sum0 = simd.add_f32s(sum0, *input);
                        }

                        let mut sum = simd.reduce_sum_f32s(sum0);

                        for input in input1 {
                            sum += input;
                        }

                        sum
                    }
                }

                Arch::new().dispatch(Impl { input: chunk })
            })
            .sum()
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .field("data_len", &self.data.borrow().len())
            .finish()
    }
}

fn bench_tensor_add(c: &mut Criterion) {
    let shape = [1000, 1000];
    let a = Tensor::zeros(&shape);
    let b = Tensor::zeros(&shape);

    c.bench_function("Tensor add 1000x1000", |bencher| {
        bencher.iter(|| {
            let _ = a.add(&b);
        });
    });

    c.bench_function("Tensor par_add 1000x1000", |bencher| {
        bencher.iter(|| {
            let _ = a.par_add(&b);
        });
    });
}

fn bench_tensor_sum(c: &mut Criterion) {
    let shape = [1000, 1000];
    let a = Tensor::zeros(&shape);

    c.bench_function("Tensor sum 1000x1000", |bencher| {
        bencher.iter(|| {
            let _ = a.sum();
        });
    });

    c.bench_function("Tensor par_sum 1000x1000", |bencher| {
        bencher.iter(|| {
            let _ = a.par_sum();
        });
    });
}

fn bench_ndarray_add(c: &mut Criterion) {
    let a = Array2::<f32>::zeros((1000, 1000));
    let b = Array2::<f32>::zeros((1000, 1000));

    c.bench_function("ndarray add 1000x1000", |bencher| {
        bencher.iter(|| {
            let _ = &a + &b;
        });
    });
}

fn bench_ndarray_sum(c: &mut Criterion) {
    let a = Array2::<f32>::zeros((1000, 1000));

    c.bench_function("ndarray sum 1000x1000", |bencher| {
        bencher.iter(|| {
            let _ = a.sum();
        });
    });
}

criterion_group!(
    benches,
    bench_tensor_add,
    bench_tensor_sum,
    bench_ndarray_add,
    bench_ndarray_sum
);

criterion_main!(benches);
