#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    #[cutile::entry(print_ir = false, unchecked_accesses = true)]
    // K is the number of columns in x and rows in y
    // BM is the number of rows in z
    // BN is the number of columns in z
    // BK is the tile size for x and y
    unsafe fn gemm_f16<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f16, { [BM, BN] }>,
        x: &Tensor<f16, { [-1, K] }>,
        y: &Tensor<f16, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);

        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);

        for i in 0i32..(K / BK) {
            let tile_x: Tile<f16, { [BM, BK] }> = part_x.load([pid.0, i]);
            let tile_y: Tile<f16, { [BK, BN] }> = part_y.load([i, pid.1]);
            let tile_x: Tile<f32, { [BM, BK] }> = convert_tile(tile_x);
            let tile_y: Tile<f32, { [BK, BN] }> = convert_tile(tile_y);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
        }
        let tile_z: Tile<f16, { [BM, BN] }> = convert_tile(tile_z);
        z.store(tile_z);
    }
}
