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

use std::{
    fmt::{self, Debug},
    iter, slice,
};

use aligned_vec::{AVec, ConstAlign, RuntimeAlign, avec, avec_rt};
use faer::traits::AddByRef;
use pulp::{Arch, WithSimd};
use rayon::prelude::*;
// use simdeez::prelude::*;

#[derive(Clone)]
pub struct Tensor {
    pub data: AVec<f32, ConstAlign<32>>,
    // pub data: AVec<f32, RuntimeAlign>,
    // pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
    pub len: usize,
    pub ndim: usize,
}

impl AddByRef for Tensor {
    type Output = Tensor;

    fn add_by_ref(&self, rhs: &Self) -> Self::Output {
        assert_eq!(self.len, rhs.len);
        let len = self.len;

        // Allocate aligned AVec result directly
        let mut result: AVec<f32, ConstAlign<32>> = avec![[32]| 0.0; len];
        // let mut result: AVec<f32, RuntimeAlign> = avec_rt![[32]| 0.0; len];
        // let mut result = vec![0.0; len];

        // Get raw slices
        // let a = &self.data;
        // let b = &rhs.data;
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
        let b_ptr = unsafe { rhs.data.as_ptr().add(rhs.offset) };

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
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product::<usize>();
        let strides = Self::compute_strides(shape);
        let data = avec![[32]| 0.0; size];
        let data_len = data.len();

        Self {
            // data: vec![0.0; size],
            // data: aligned_vec::avec_rt![[32]| 0.0; size],
            data,
            shape: shape.to_vec(),
            strides,
            offset: 0,
            len: data_len,
            ndim: shape.len(),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let data_len = data.len();
        assert_eq!(data_len, shape.iter().product::<usize>());
        let strides = Self::compute_strides(shape);
        let data = AVec::from_iter(32, data);

        Self {
            data,
            shape: shape.to_vec(),
            strides,
            offset: 0,
            len: data_len,
            ndim: shape.len(),
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        assert_eq!(self.len, new_shape.iter().product());

        Tensor {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            strides: Self::compute_strides(new_shape),
            offset: self.offset,
            len: self.len,
            ndim: new_shape.len(),
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
        assert_eq!(idx.len(), self.ndim);
        // idx.par_iter().zip(self.strides.iter()).
        self.offset
            + idx
                .iter()
                .zip(self.strides.iter())
                .map(|(i, s)| i * s)
                .sum::<usize>()
    }

    pub fn get(&self, idx: &[usize]) -> f32 {
        self.data[self.index(idx)]
    }

    pub fn set(&mut self, idx: &[usize], value: f32) {
        let i = self.index(idx);
        self.data[i] = value;
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
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

    pub fn sum(&self) -> f32 {
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

    pub fn mean(&self) -> f32 {
        self.sum() / self.len as f32
    }

    pub fn swapaxes(&self, axis1: usize, axis2: usize) -> Tensor {
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

    pub fn narrow(&self, axis: usize, start: usize, end: usize) -> Tensor {
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

    /// Parallel sum using SIMD + Rayon
    pub fn par_sum(&self) -> f32 {
        self.data
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
            .field("data_len", &self.len)
            .finish()
    }
}

// pub fn par_add(&self, other: &Tensor) -> Tensor {
//     assert_eq!(self.data.len(), other.data.len());
//     let len = self.data.len();
//
//     // Allocate aligned AVec result directly
//     let mut result: AVec<f32, ConstAlign<32>> = avec![[32]| 0.0; len];
//
//     // Tune chunk size: larger chunks reduce scheduling overhead for small workloads
//     let chunk = 16384usize.min(len.max(1024)); // heuristic
//
//     // Work on raw pointers to avoid bounds checks
//     let a_ptr = self.data.as_ptr();
//     let b_ptr = other.data.as_ptr();
//     let out_ptr = result.as_mut_ptr();
//
//     unsafe {
//         // rayon parallel loop operating on indexes
//         (0..len).into_par_iter().with_max_len(chunk).for_each(|i| {
//             // simple element-wise add — minimal overhead
//             // Using safe indexing here would reintroduce checks; use unsafe raw loads/stores
//             unsafe {
//                 let ai = *a_ptr.add(i);
//                 let bi = *b_ptr.add(i);
//                 *out_ptr.add(i) = ai + bi;
//             }
//         });
//     }
//
//     Tensor {
//         data: result,
//         shape: self.shape.clone(),
//         strides: self.strides.clone(),
//         offset: self.offset,
//     }
// }

// pub fn add(&self, other: &Tensor) -> Tensor {
//     assert_eq!(self.data.len(), other.data.len());
//     let mut result = vec![0.0f32; self.data.len()];
//
//     struct Impl<'a> {
//         out: &'a mut [f32],
//         a: &'a [f32],
//         b: &'a [f32],
//     }
//
//     impl WithSimd for Impl<'_> {
//         type Output = ();
//
//         #[inline(always)]
//         fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
//             let Self { out, a, b } = self;
//
//             let (out0, out1) = S::as_mut_simd_f32s(out);
//             let (a0, a1) = S::as_simd_f32s(a);
//             let (b0, b1) = S::as_simd_f32s(b);
//
//             for (out, (a, b)) in iter::zip(out0, iter::zip(a0, b0)) {
//                 *out = simd.add_f32s(*a, *b);
//             }
//
//             for (out, (a, b)) in iter::zip(out1, iter::zip(a1, b1)) {
//                 *out = *a + *b;
//             }
//         }
//     }
//
//     Arch::new().dispatch(Impl {
//         out: &mut result,
//         a: &self.data,
//         b: &other.data,
//     });
//
//     Tensor::from_vec(result, &self.shape)
// }

// pub fn sum(&self) -> f32 {
//     self.data.iter().sum()
// }
//
// pub fn par_add(&self, other: &Tensor) -> Tensor {
//     assert_eq!(self.data.len(), other.data.len());
//     let len = self.data.len();
//     let mut result = vec![0.0f32; len];
//
//     result
//         .par_chunks_mut(1024)
//         .zip(self.data.par_chunks(1024))
//         .zip(other.data.par_chunks(1024))
//         .for_each(|((out, a), b)| {
//             struct Impl<'a> {
//                 out: &'a mut [f32],
//                 a: &'a [f32],
//                 b: &'a [f32],
//             }
//
//             impl WithSimd for Impl<'_> {
//                 type Output = ();
//
//                 #[inline(always)]
//                 fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
//                     let Self { out, a, b } = self;
//
//                     let (out0, out1) = S::as_mut_simd_f32s(out);
//                     let (a0, a1) = S::as_simd_f32s(a);
//                     let (b0, b1) = S::as_simd_f32s(b);
//
//                     for (out, (a, b)) in iter::zip(out0, iter::zip(a0, b0)) {
//                         *out = simd.add_f32s(*a, *b);
//                     }
//
//                     for (out, (a, b)) in iter::zip(out1, iter::zip(a1, b1)) {
//                         *out = *a + *b;
//                     }
//                 }
//             }
//
//             Arch::new().dispatch(Impl { out, a, b });
//         });
//
//     Tensor::from_vec(result, &self.shape)
// }
