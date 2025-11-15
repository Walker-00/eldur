use eldur::Tensor;
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();

    // Shape of the tensor
    let shape = [2; 2];
    let len = shape.iter().product::<usize>();

    // Generate random values for tensor a
    let a_values: Vec<f32> = (0..len).map(|_| rng.gen_range(0.0..10.0)).collect();
    let a = Tensor::from_vec(a_values, &shape);

    // Generate random values for tensor b
    let b_values: Vec<f32> = (0..len).map(|_| rng.gen_range(0.0..10.0)).collect();
    let b = Tensor::from_vec(b_values, &shape);

    println!("Tensor a: {:?}", &a.data);
    println!("Tensor b: {:?}", &b.data);

    let c = a.add(&b);
    println!("Tensor c (a + b): {:?}", c.data);
}
