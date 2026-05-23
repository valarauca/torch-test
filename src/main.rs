
use tch::{Cuda};

pub fn main() {
    println!("Cuda device count: {}", Cuda::device_count());
    println!("Cuda avaliable: {}", Cuda::is_available());
    println!("Cudnn is available: {}", Cuda::cudnn_is_available());
}
