// use std::ops::{Add, Deref, Mul};
//
// use ndarray::{ArrayD, IxDyn};
// use num_traits::{One, Zero};
//
// pub struct Tensor<T> {
//     pub data: ArrayD<T>,
// }
//
// impl<T> Deref for Tensor<T> {
//     type Target = ArrayD<T>;
//
//     fn deref(&self) -> &Self::Target {
//         &self.data
//     }
// }
//
// impl<T> Tensor<T>
// where
//     T: Clone + Zero + One,
// {
//     pub fn from_array(data: ArrayD<T>) -> Self {
//         Self { data }
//     }
//
//     pub fn zeros(shape: &[usize]) -> Self {
//         Self {
//             data: ArrayD::zeros(IxDyn(shape)),
//         }
//     }
//
//     pub fn ones(shape: &[usize]) -> Self {
//         Self {
//             data: ArrayD::ones(IxDyn(shape)),
//         }
//     }
//
//     pub fn shape(&self) -> &[usize] {
//         self.data.shape()
//     }
//
//     pub fn get(&self, index: &[usize]) -> Option<&T> {
//         self.data.get(index)
//     }
//
//     pub fn set(&mut self, index: &[usize], value: T) {
//         if let Some(elem) = self.data.get_mut(index) {
//             *elem = value;
//         }
//     }
// }
//
// impl<T> Add for Tensor<T>
// where
//     T: Clone + Zero + One,
// {
//     type Output = Self;
//
//     fn add(self, rhs: Self) -> Self::Output {
//         Self {
//             data: self.data + rhs.data,
//         }
//     }
// }
//
// impl<T> Mul for Tensor<T>
// where
//     T: Clone + Zero + One,
// {
//     type Output = Self;
//
//     fn mul(self, rhs: Self) -> Self::Output {
//         Self {
//             data: self.data * rhs.data,
//         }
//     }
// }

use std::ops::Deref;

use ndarray::{Array, ArrayD};
use rand::Rng;

#[derive(Debug, Clone)]
struct Tensor<T> {
    data: ArrayD<T>,
}

impl<T> Tensor<T> {
    fn data(data: ArrayD<T>) -> Self {
        Self { data }
    }

    fn random() -> Self {
        let mut rng = rand::rng();

        let num_dims = rng.random_range(1..=4);

        let shape: Vec<usize> = (0..num_dims).map(|_| rng.random_range(1..=10)).collect();

        let data = ArrayD::
    }
}

impl<T> Deref for Tensor<T> {
    type Target = ArrayD<T>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

fn main() {
    println!("Hello, world!");
}
