/*
 * Later Todo List To Improve
 *
 * For The Performance: CGT_6917e073-f874-8003-89e6-a12a3f041e17
 * Make The very performance Aligned Vector Storage Backend for Tensor with Arc + Copy-On-Write(COW)
 * Add the Cached values to the tensor like computing the dims and len
 * Make The Optionales Methods with par_<methods_name> for Utilizing the Parallel Computing
 *
 * For The Compatibility: GMN_4e8f478ba9002916
 * Add the offset options for better Compatibility with others frameworks
 */

use eldur::Tensor;
use faer::Mat;
use ndarray::{Array2, Axis};
use rayon::prelude::*; // Required for ndarray parallel sum

// ---
// 1. SET A LARGE SIZE
// A 2x2 array (4 elements) is too small. Overhead will dominate.
// Let's use 1024x1024 (1,048,576 elements) to actually test SIMD and Rayon.
// ---
const ROWS: usize = 1024;
const COLS: usize = 1024;
const TOTAL_ELEMENTS: usize = ROWS * COLS;

const SHAPE: [usize; 2] = [ROWS, COLS];
const NDARRAY_SHAPE: (usize, usize) = (ROWS, COLS);

// ---
// 2. USE THE BENCHER CORRECTLY
// We must separate setup (creating the tensors) from the code being benchmarked.
// `bencher.bench_local(|| ...)` is the standard way to do this in divan.
// We also use `std::hint::black_box` to prevent the compiler from
// optimizing away the operation.
//
// We also use non-zero data (e.g., `vec![1.0f32; ...]`)
// as benchmarking on `zeros` can sometimes be misleading.
// ---

#[divan::bench(sample_size = 100, sample_count = 10)]
fn bench_tensor_add(bencher: divan::Bencher) {
    let a = Tensor::from_vec(vec![1.0f32; TOTAL_ELEMENTS], &SHAPE);
    let b = Tensor::from_vec(vec![1.0f32; TOTAL_ELEMENTS], &SHAPE);
    bencher.bench_local(|| {
        std::hint::black_box(a.add(&b));
    });
}

#[divan::bench(sample_size = 100, sample_count = 10)]
fn bench_ndarray_add(bencher: divan::Bencher) {
    let a = Array2::<f32>::from_elem(NDARRAY_SHAPE, 1.0);
    let b = Array2::<f32>::from_elem(NDARRAY_SHAPE, 1.0);
    bencher.bench_local(|| {
        std::hint::black_box(&a + &b);
    });
}

#[divan::bench(sample_size = 100, sample_count = 10)]
fn bench_faer_add(bencher: divan::Bencher) {
    let a = Mat::<f32>::from_fn(ROWS, COLS, |_, _| 1.0);
    let b = Mat::<f32>::from_fn(ROWS, COLS, |_, _| 1.0);
    bencher.bench_local(|| {
        std::hint::black_box(&a + &b);
    });
}

// ---
// SINGLE-THREADED SUMMATION
// ---

#[divan::bench(sample_size = 100, sample_count = 10)]
fn bench_tensor_sum(bencher: divan::Bencher) {
    let a = Tensor::from_vec(vec![1.0f32; TOTAL_ELEMENTS], &SHAPE);
    bencher.bench_local(|| {
        std::hint::black_box(a.sum());
    });
}

#[divan::bench(sample_size = 100, sample_count = 10)]
fn bench_ndarray_sum(bencher: divan::Bencher) {
    let a = Array2::<f32>::from_elem(NDARRAY_SHAPE, 1.0);
    bencher.bench_local(|| {
        // .sum() on ndarray is single-threaded
        std::hint::black_box(a.sum());
    });
}

// ---
// MULTI-THREADED (PARALLEL) SUMMATION
// ---

#[divan::bench(sample_size = 100, sample_count = 10)]
fn bench_tensor_sum_parallel(bencher: divan::Bencher) {
    let a = Tensor::from_vec(vec![1.0f32; TOTAL_ELEMENTS], &SHAPE);
    bencher.bench_local(|| {
        std::hint::black_box(a.par_sum());
    });
}

#[divan::bench(sample_size = 100, sample_count = 10)]
fn bench_ndarray_sum_parallel(bencher: divan::Bencher) {
    let a = Array2::<f32>::from_elem(NDARRAY_SHAPE, 1.0);
    bencher.bench_local(|| {
        // To get a parallel sum in ndarray, you must use rayon explicitly
        std::hint::black_box(a.par_iter().sum::<f32>());
    });
}

#[divan::bench(sample_size = 100, sample_count = 10)]
fn bench_faer_sum_parallel(bencher: divan::Bencher) {
    let a = Mat::<f32>::from_fn(ROWS, COLS, |_, _| 1.0);
    bencher.bench_local(|| {
        // faer's .sum() is parallel by default if compiled with the "rayon" feature
        std::hint::black_box(a.sum());
    });
}

fn main() {
    // IMPORTANT: Run this with `cargo run --release`
    divan::main();
}

