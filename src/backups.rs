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
