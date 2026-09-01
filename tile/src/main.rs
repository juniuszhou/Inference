mod kernels;

use cutile::prelude::*;

#[cutile::module]
mod kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const B: i32>(
        z: &mut Tensor<f32, { [B] }>, // Tensor is GPU global memory
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like(x, z); // Tile is GPU shared memory within the same SM
        let ty = load_tile_like(y, z);
        z.store(tx + ty);
    }
}

// Architectures below sm_80 (for example sm_70 and sm_75) are out of scope
#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), Error> {
    let x = api::ones::<f32>(&[1024]);
    let y = api::ones::<f32>(&[1024]);
    let z = api::zeros::<f32>(&[1024]).partition([128]);

    let (_z, _x, _y) = kernel::add(z, x, y).sync()?;
    let z = _z.unpartition().to_host_vec().await?;
    println!("len: {}", z.len());
    println!("first 10: {:?}", &z[..10.min(z.len())]);
    println!("all: {:?}", z);
    Ok(())
}
