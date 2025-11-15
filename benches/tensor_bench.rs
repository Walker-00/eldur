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

use eldur::Tensor;
use ndarray::Array2;

// const SHAPE: [usize; 2] = [2, 2];
const NDARRAY_SHAPE: (usize, usize) = (2, 2);

#[divan::bench]
fn bench_tensor_add() {
    let a = Tensor::from_vec(vec![3.2, 1.7, 9.4, 5.1], &[2; 2]);
    let b = Tensor::from_vec(vec![4.8, 2.3, 1.9, 7.6], &[2; 2]);
    let _ = a.add(&b);
}

#[divan::bench]
fn bench_tensor_add_parallel() {
    let a = Tensor::from_vec(vec![3.2, 1.7, 9.4, 5.1], &[2; 2]);
    let b = Tensor::from_vec(vec![4.8, 2.3, 1.9, 7.6], &[2; 2]);
    a.par_add(&b);
}

#[divan::bench]
fn bench_tensor_sum() {
    let a = Tensor::from_vec(vec![3.2, 1.7, 9.4, 5.1], &[2; 2]);
    let _ = a.sum();
}

#[divan::bench]
fn bench_tensor_sum_parallel() {
    let a = Tensor::from_vec(vec![3.2, 1.7, 9.4, 5.1], &[2; 2]);
    let _ = a.par_sum();
}

#[divan::bench]
fn bench_ndarray_add() {
    let a = Array2::<f32>::zeros(NDARRAY_SHAPE);
    let b = Array2::<f32>::zeros(NDARRAY_SHAPE);

    let _ = &a + &b;
}

#[divan::bench]
fn bench_ndarray_sum() {
    let a = Array2::<f32>::zeros(NDARRAY_SHAPE);

    let _ = a.sum();
}

fn main() {
    divan::main();
}
