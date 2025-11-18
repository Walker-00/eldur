use eldur::Tensor;
use faer::{Mat, traits::AddByRef};
use ndarray::Array2;
use rayon::prelude::*; // Required for ndarray parallel sum

// ---
// 1. SET MULTIPLE SIZES
// We define a list of side-lengths (N).
// The benchmark will run on N x N arrays.
// This will test:
// - 2x2 (4 elements) - Your original "slow" case
// - 64x64 (4,096 elements) - Small
// - 256x256 (65,536 elements) - Medium
// - 1024x1024 (1,048,576 elements) - Large
// ---
const SIDES: [usize; 6] = [2, 64, 256, 1024, 2048, 4096];
// const SIDES: [usize; 3] = [2, 64, 256];

// ---
// 2. HELPER FUNCTION
// Helper to create a shape [N, N] and total elements.
// ---
fn setup(n: usize) -> ([usize; 2], (usize, usize), usize) {
    let shape = [n, n];
    let ndarray_shape = (n, n);
    let total_elements = n * n;
    (shape, ndarray_shape, total_elements)
}

// ---
// ADDITION BENCHMARKS
// ---

#[divan::bench(args = SIDES, sample_size = 100, sample_count = 100)]
fn bench_tensor_add(bencher: divan::Bencher, n: usize) {
    let (shape, _, total_elements) = setup(n);
    let a = Tensor::from_vec(vec![1.0f32; total_elements], &shape);
    let b = Tensor::from_vec(vec![1.0f32; total_elements], &shape);
    bencher.bench_local(|| {
        std::hint::black_box(a.add(&b));
    });
}

#[divan::bench(args = SIDES, sample_size = 100, sample_count = 100)]
fn bench_ndarray_add(bencher: divan::Bencher, n: usize) {
    let (_, ndarray_shape, _) = setup(n);
    let a = Array2::<f32>::from_elem(ndarray_shape, 1.0);
    let b = Array2::<f32>::from_elem(ndarray_shape, 1.0);
    bencher.bench_local(|| {
        std::hint::black_box(a.add_by_ref(&b));
    });
}

#[divan::bench(args = SIDES, sample_size = 100, sample_count = 100)]
fn bench_faer_add(bencher: divan::Bencher, n: usize) {
    let (rows, cols) = (n, n);
    let a = Mat::<f32>::from_fn(rows, cols, |_, _| 1.0);
    let b = Mat::<f32>::from_fn(rows, cols, |_, _| 1.0);
    bencher.bench_local(|| {
        std::hint::black_box(a.add_by_ref(&b));
    });
}

// ---
// SINGLE-THREADED SUMMATION
// ---

#[divan::bench(args = SIDES, sample_size = 100, sample_count = 100)]
fn bench_tensor_sum(bencher: divan::Bencher, n: usize) {
    let (shape, _, total_elements) = setup(n);
    let a = Tensor::from_vec(vec![1.0f32; total_elements], &shape);
    bencher.bench_local(|| {
        std::hint::black_box(a.sum());
    });
}

#[divan::bench(args = SIDES, sample_size = 100, sample_count = 100)]
fn bench_ndarray_sum(bencher: divan::Bencher, n: usize) {
    let (_, ndarray_shape, _) = setup(n);
    let a = Array2::<f32>::from_elem(ndarray_shape, 1.0);
    bencher.bench_local(|| {
        std::hint::black_box(a.sum());
    });
}

// ---
// MULTI-THREADED (PARALLEL) SUMMATION
// ---

#[divan::bench(args = SIDES, sample_size = 100, sample_count = 100)]
fn bench_tensor_sum_parallel(bencher: divan::Bencher, n: usize) {
    let (shape, _, total_elements) = setup(n);
    let a = Tensor::from_vec(vec![1.0f32; total_elements], &shape);
    bencher.bench_local(|| {
        std::hint::black_box(a.par_sum());
    });
}

#[divan::bench(args = SIDES, sample_size = 100, sample_count = 100)]
fn bench_ndarray_sum_parallel(bencher: divan::Bencher, n: usize) {
    let (_, ndarray_shape, _) = setup(n);
    let a = Array2::<f32>::from_elem(ndarray_shape, 1.0);
    bencher.bench_local(|| {
        std::hint::black_box(a.par_iter().sum::<f32>());
    });
}

#[divan::bench(args = SIDES, sample_size = 100, sample_count = 100)]
fn bench_faer_sum_parallel(bencher: divan::Bencher, n: usize) {
    let (rows, cols) = (n, n);
    let a = Mat::<f32>::from_fn(rows, cols, |_, _| 1.0);
    bencher.bench_local(|| {
        std::hint::black_box(a.sum());
    });
}

fn main() {
    divan::main();
}
