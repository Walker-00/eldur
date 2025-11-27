use aligned_vec::{AVec, ConstAlign, avec};
use pulp::{Arch, WithSimd};
use rayon::prelude::*;
use std::{
    fmt::{self, Debug},
    iter, slice,
};

use crate::{
    Tensor,
    traits::{ParTensorOps, TensorOps},
};

impl TensorOps for Tensor {
    type SeqTensor = Tensor;
    fn add(&self, other: &Self::SeqTensor) -> Self::SeqTensor {
        assert_eq!(self.len, other.len);
        let len = self.len;

        // Allocate aligned AVec result directly
        let mut result: AVec<f32, ConstAlign<32>> = avec![[32]| 0.0; len];
        // let mut result: AVec<f32, RuntimeAlign> = avec_rt![[32]| 0.0; len];
        // let mut result = vec![0.0; len];

        // Get raw slices
        // let a = &self.data;
        // let b = &other.data;
        // let out = &mut result;

        // Use the same WithSimd dispatch but iterate over slices, and use unsafe to avoid bounds checks
        struct Impl<'a> {
            out: &'a mut [f32],
            a: &'a [f32],
            b: &'a [f32],
        }

        impl WithSimd for Impl<'_> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let Impl { out, a, b } = self;

                let (out0, out1) = S::as_mut_simd_f32s(out);
                let (a0, a1) = S::as_simd_f32s(a);
                let (b0, b1) = S::as_simd_f32s(b);

                for (out_simd, (a_simd, b_simd)) in iter::zip(out0, iter::zip(a0, b0)) {
                    *out_simd = simd.add_f32s(*a_simd, *b_simd);
                }

                // Handle tail (scalar part)
                for (out_s, (a_s, b_s)) in iter::zip(out1, iter::zip(a1, b1)) {
                    *out_s = *a_s + *b_s;
                }
            }
        }

        let out_ptr = result.as_mut_ptr();

        let a_ptr = unsafe { self.data.as_ptr().add(self.offset) };
        let b_ptr = unsafe { other.data.as_ptr().add(other.offset) };

        let a_slice = unsafe { slice::from_raw_parts(a_ptr, len) };
        let b_slice = unsafe { slice::from_raw_parts(b_ptr, len) };
        let out_slice = unsafe { slice::from_raw_parts_mut(out_ptr, len) };

        Arch::new().dispatch(Impl {
            out: out_slice,
            a: a_slice,
            b: b_slice,
        });

        Tensor {
            data: result,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            len: len,
            ndim: self.ndim,
        }
    }

    fn sum(&self) -> f32 {
        struct Impl<'a> {
            input: &'a [f32],
        }

        impl WithSimd for Impl<'_> {
            type Output = f32;

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                let (input0, input1) = S::as_simd_f32s(self.input);
                let (input04, input01) = pulp::as_arrays::<4, _>(input0);

                let mut sums = [simd.splat_f32s(0.0); 4];

                // let mut sum0 = simd.splat_f32s(0.0);
                // let mut sum1 = simd.splat_f32s(0.0);
                // let mut sum2 = simd.splat_f32s(0.0);
                // let mut sum3 = simd.splat_f32s(0.0);

                // for [input0, input1, input2, input3] in input04 {
                //     sum0 = simd.add_f32s(sum0, *input0);
                //     sum1 = simd.add_f32s(sum1, *input1);
                //     sum2 = simd.add_f32s(sum2, *input2);
                //     sum3 = simd.add_f32s(sum3, *input3);
                // }
                //
                // sum0 = simd.add_f32s(sum0, sum1);
                // sum2 = simd.add_f32s(sum2, sum3);
                //
                // sum0 = simd.add_f32s(sum0, sum2);
                //
                // for input in input01 {
                //     sum0 = simd.add_f32s(sum0, *input);
                // }
                //
                // let mut sum = simd.reduce_sum_f32s(sum0);

                for [chunk0, chunk1, chunk2, chunk3] in input04 {
                    sums[0] = simd.add_f32s(sums[0], *chunk0);
                    sums[1] = simd.add_f32s(sums[1], *chunk1);
                    sums[2] = simd.add_f32s(sums[2], *chunk2);
                    sums[3] = simd.add_f32s(sums[3], *chunk3);
                }

                sums[0] = simd.add_f32s(sums[0], sums[1]);
                sums[2] = simd.add_f32s(sums[2], sums[3]);
                sums[0] = simd.add_f32s(sums[0], sums[2]);

                for &chunk in input01 {
                    sums[0] = simd.add_f32s(sums[0], chunk);
                }

                // let mut sum = simd.reduce_sum_f32s(sum0);
                let mut total = simd.reduce_sum_f32s(sums[0]);

                for &val in input1 {
                    total += val;
                }

                total
            }
        }
        Arch::new().dispatch(Impl { input: &self.data })
    }

    fn mean(&self) -> f32 {
        self.sum() / self.len as f32
    }

    fn swapaxes(&self, axis1: usize, axis2: usize) -> Tensor {
        assert!(
            axis1 < self.ndim || axis2 < self.ndim,
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
            len: self.len,
            ndim: self.ndim,
        }
    }

    fn narrow(&self, axis: usize, start: usize, end: usize) -> Tensor {
        assert!(axis < self.ndim, "Narrow axis index out of bounds.");
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
            len: self.len,
            ndim: self.ndim,
        }
    }
}

impl ParTensorOps for Tensor {
    type ParTensor = Tensor;

    fn par_add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.data.len(), other.data.len());
        let len = self.data.len();
        let mut result = vec![0.0f32; len];

        result
            .par_chunks_mut(1024)
            .zip(self.data.par_chunks(1024))
            .zip(other.data.par_chunks(1024))
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

    fn par_sum(&self) -> f32 {
        self.data.par_iter().sum()
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .field("data_len", &self.len)
            .finish()
    }
}
