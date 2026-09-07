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
        }
        let tile_z: Tile<f16, { [BM, BN] }> = convert_tile(tile_z);
        z.store(tile_z);
    }

    #[cutile::entry()]
    fn add_1d<const B: i32>(
        // An entry *output* cannot be Tensor<[-1]>: the launcher checks that
        z: &mut Tensor<f32, { [B] }>, // Tensor is GPU global memory
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        // tx with the same shape as z, it is not just the shape, also including the partition position.
        let tx: Tile<f32, { [B] }> = load_tile_like(x, z); // Tile is GPU shared memory within the same SM
        let ty: Tile<f32, { [B] }> = load_tile_like(y, z);
        z.store(tx + ty);
    }

    #[cutile::entry()]
    fn sum_1d(
        z: &mut Tensor<f32, { [1] }>, // single value as output (1-element tile)
        x: &Tensor<f32, { [-1] }>,
    ) {
        let part_x = x.partition(const_shape![32]);
        // get the number of tiles in the first dimension
        let n = num_tiles(&part_x, 0);
        let mut acc: Tile<f32, { [] }> = scalar_to_tile(0.0f32);
        for i in 0i32..n {
            let tile: Tile<f32, { [32] }> = part_x.load([i]);
            // reduce the tile to a scalar in the first dimension
            let tile_sum: Tile<f32, { [] }> = reduce_sum(tile, 0i32);
            acc = acc + tile_sum;
        }
        // from a scalar to a 1-element tile, for storage later in z
        let acc1: Tile<f32, { [1] }> = acc.reshape(const_shape![1]);
        z.store(acc1);
    }

    #[cutile::entry()]
    fn sum_1d_p(
        z: &mut Tensor<f32, { [1] }>, // single value as output (1-element tile)
        x: &Tensor<f32, { [-1] }>,
    ) {
        let part_x = x.partition(const_shape![32]);
        let n = num_tiles(&part_x, 0);
        let mut acc: Tile<f32, { [] }> = scalar_to_tile(0.0f32);
        for i in 0i32..n {
            let tile: Tile<f32, { [32] }> = part_x.load([i]);
            let tile_sum: Tile<f32, { [] }> = reduce_sum(tile, 0i32);
            acc = acc + tile_sum;
        }
        let acc1: Tile<f32, { [1] }> = acc.reshape(const_shape![1]);
        z.store(acc1);
    }

    // 二阶段规约 — 第一阶段：每 CTA 计算一个 Tile[32] 的和，写入 partial[pid]
    #[cutile::entry()]
    fn sum_stage1(partial: &mut Tensor<f32, { [1] }>, x: &Tensor<f32, { [-1] }>) {
        let pid = get_tile_block_id().0;
        let part_x = x.partition(const_shape![32]);
        let tile: Tile<f32, { [32] }> = part_x.load([pid]);
        let tile_sum: Tile<f32, { [] }> = reduce_sum(tile, 0i32);
        let tile_sum1: Tile<f32, { [1] }> = tile_sum.reshape(const_shape![1]);
        partial.store(tile_sum1);
    }

    // 二阶段规约 — 第二阶段：单 CTA 归约 partial[n] -> z[1]
    #[cutile::entry()]
    fn sum_stage2(z: &mut Tensor<f32, { [1] }>, partial: &Tensor<f32, { [-1] }>) {
        let part_p = partial.partition(const_shape![32]);
        let n = num_tiles(&part_p, 0);
        let mut acc: Tile<f32, { [] }> = scalar_to_tile(0.0f32);
        for i in 0i32..n {
            let tile: Tile<f32, { [32] }> = part_p.load([i]);
            let tile_sum: Tile<f32, { [] }> = reduce_sum(tile, 0i32);
            acc = acc + tile_sum;
        }
        let acc1: Tile<f32, { [1] }> = acc.reshape(const_shape![1]);
        z.store(acc1);
    }

    // the second add_1d is for the case that the shape of x and y is not the same as z.
    // it is clearer to use the partition shape of x and y.
    #[cutile::entry()]
    fn add_1d_2<const B: i32>(
        // An entry *output* cannot be Tensor<[-1]>: the launcher checks that
        z: &mut Tensor<f32, { [B] }>, // Tensor is GPU global memory
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_x = x.partition(const_shape![B]);
        let part_y = y.partition(const_shape![B]);
        // tx with the same shape as z
        let tx: Tile<f32, { [B] }> = part_x.load([pid.0]); // Tile is GPU shared memory within the same SM
        let ty: Tile<f32, { [B] }> = part_y.load([pid.0]);
        z.store(tx + ty);
    }

    #[cutile::entry()]
    fn add_2d<const BM: i32, const BN: i32, const BK: i32>(
        // An entry *output* cannot be Tensor<[-1]>: the launcher checks that
        z: &mut Tensor<f32, { [BM, BN] }>, // Tensor is GPU global memory
        x: &Tensor<f32, { [-1, BK] }>,
        y: &Tensor<f32, { [-1, BK] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_x = x.partition(const_shape![BM, BN]);
        let part_y = y.partition(const_shape![BM, BN]);

        for i in 0i32..(BK / BN) {
            let tile_x: Tile<f32, { [BM, BN] }> = part_x.load([pid.0, i]);
            let tile_y: Tile<f32, { [BM, BN] }> = part_y.load([pid.0, i]);
            let tile_z: Tile<f32, { [BM, BN] }> = tile_x + tile_y;
            z.store(tile_z);
        }
    }

    unsafe fn load_f16_ptr(
        ptrs: &Tensor<i64, { [-1] }>,
        group_id: i32,
    ) -> PointerTile<*mut f16, { [] }> {
        let one_shape: Shape<{ [1] }> = Shape::<{ [1] }> { dims: &[] };
        // partition with single dimension and just one element in each group/partition
        let ptr_part: Partition<i64, { [1] }> = ptrs.partition(one_shape);
        let ptr_int: Tile<i64, { [1] }> = ptr_part.load([group_id]);
        let ptr_int: Tile<i64, { [] }> = ptr_int.reshape(const_shape![]);
        // convert the integer to a pointer
        let ptr: PointerTile<*mut f16, { [] }> = int_to_ptr(ptr_int);
        // assume the pointer is divisible by 16, make sure the pointer is aligned to 2 bytes
        unsafe { assume_div_by::<_, 16>(ptr) }
    }

    #[cutile::entry()]
    unsafe fn copy_via_load_f16_ptr<const B: i32>(
        z: &mut Tensor<f16, { [B] }>, // output tensor
        ptrs: &Tensor<i64, { [-1] }>,
        group_id: i32, // the group id of the pointers
        len: i32,      // the length of the pointers
    ) {
        let pid = get_tile_block_id().0;
        let ptr = unsafe { load_f16_ptr(ptrs, group_id) };
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let src: Tensor<f16, { [-1] }> =
            unsafe { make_tensor_view(ptr, shape, strides, new_token_unordered()) };
        z.store(src.partition(const_shape![B]).load([pid]));
    }

    #[cutile::entry(print_ir=false,
        optimization_hints = (
          sm_120 = (occupancy=1, max_divisibility=16,),
        ))]
    fn silu_mul_vec_f16<const S: [i32; 1]>(
        out: &mut Tensor<f16, S>,
        gate: &Tensor<f16, { [-1] }>,
        up: &Tensor<f16, { [-1] }>,
    ) {
        let gate_f16 = load_tile_like(gate, out);
        let up_f16 = load_tile_like(up, out);
        let gate_f32: Tile<f32, S> = convert_tile(gate_f16);
        let up_f32: Tile<f32, S> = convert_tile(up_f16);
        let one: Tile<f32, S> = constant(1.0f32, out.shape());
        let zero: Tile<f32, S> = constant(0.0f32, out.shape());
        let neg_gate: Tile<f32, S> = zero - gate_f32;
        let exp_neg_gate: Tile<f32, S> = exp(neg_gate);
        let denom: Tile<f32, S> = one + exp_neg_gate;
        let sigmoid: Tile<f32, S> = true_div(one, denom);
        let y: Tile<f32, S> = sigmoid * gate_f32 * up_f32;
        let y_f16: Tile<f16, S> = convert_tile(y);
        out.store(y_f16);
    }

    #[cutile::entry(print_ir=false,
        unchecked_accesses=true,
        optimization_hints = (
          sm_100 = (max_divisibility=8,),
          sm_120 = (max_divisibility=8,),
        ))]
    unsafe fn rms_norm_f16<const N: i32, const BLOCK_SIZE: i32>(
        x: &Tensor<f16, { [-1, N] }>,
        w: &Tensor<f16, { [N] }>,
        out: &mut Tensor<f16, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let x_part: Partition<f16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            rms = rms + tx * tx;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);

        let inv_rms: f32 = rms / n + eps;
        let inv_rms: Tile<f32, { [] }> = rsqrt(scalar_to_tile(inv_rms), ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

        let w_part: Partition<f16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(tile_shape) };
        for j in 0i32..num_tiles {
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw_f16: Tile<f16, { [1, BLOCK_SIZE] }> = w_part.load([j]).reshape(tile_shape);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tw_f16);
            let tout: Tile<f32, { [1, BLOCK_SIZE] }> = tx * inv_rms * tw;
            let tout_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(tout);
            unsafe { out_part.store(tout_f16, [0i32, j]) };
        }
    }

    #[cutile::entry(print_ir=false,
        optimization_hints = (
          sm_120 = (occupancy=1, max_divisibility=16,),
        ))]
    fn argmax_blocks_f16<const BLOCK_SIZE: i32>(
        logits: &Tensor<f16, { [-1] }>,
        block_max: &mut Tensor<f32, { [1] }>,
        block_idx: &mut Tensor<u32, { [1] }>,
        // length of the logits
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let block = pid.0;

        let logits_part = logits.partition(const_shape![BLOCK_SIZE]);
        let logits_f16: Tile<f16, { [BLOCK_SIZE] }> = logits_part.load([block]);
        let logits: Tile<f32, { [BLOCK_SIZE] }> = convert_tile(logits_f16);

        let base: i32 = block * BLOCK_SIZE;
        // repeat the base value for each block
        let base_tile: Tile<i32, { [BLOCK_SIZE] }> = base.broadcast(const_shape![BLOCK_SIZE]);
        let offs: Tile<i32, { [BLOCK_SIZE] }> = iota(const_shape![BLOCK_SIZE]);
        let indices: Tile<i32, { [BLOCK_SIZE] }> = base_tile + offs;

        let len_tile: Tile<i32, { [BLOCK_SIZE] }> = len.broadcast(const_shape![BLOCK_SIZE]);
        // compare the indices with the length, then we can ignore the padding values
        let valid: Tile<bool, { [BLOCK_SIZE] }> = lt_tile(indices, len_tile);

        // mask the logits that are out of range
        let mask_mag: Tile<f32, { [BLOCK_SIZE] }> = constant(1.0e30f32, const_shape![BLOCK_SIZE]);
        let zero: Tile<f32, { [BLOCK_SIZE] }> = constant(0.0f32, const_shape![BLOCK_SIZE]);
        // it is small enought as the input type is f16, so the overflow is not a problem
        let neg_inf: Tile<f32, { [BLOCK_SIZE] }> = zero - mask_mag;
        let masked_logits: Tile<f32, { [BLOCK_SIZE] }> = select(valid, logits, neg_inf);

        let max_tile: Tile<f32, { [1] }> = reduce_max(masked_logits, 0i32);
        let max_scalar: f32 = tile_to_scalar(max_tile.reshape(const_shape![]));

        let max_bcast: Tile<f32, { [BLOCK_SIZE] }> = max_scalar.broadcast(const_shape![BLOCK_SIZE]);
        let is_max: Tile<bool, { [BLOCK_SIZE] }> = eq_tile(masked_logits, max_bcast);
        let invalid_idx: i32 = len + 1i32;
        let invalid_idx: Tile<i32, { [BLOCK_SIZE] }> =
            invalid_idx.broadcast(const_shape![BLOCK_SIZE]);
        let candidate_idx: Tile<i32, { [BLOCK_SIZE] }> = select(is_max, indices, invalid_idx);
        let idx_tile: Tile<i32, { [1] }> = reduce_min(candidate_idx, 0i32);
        let idx_scalar: i32 = tile_to_scalar(idx_tile.reshape(const_shape![]));

        let out_max_scalar: Tile<f32, { [] }> = scalar_to_tile(max_scalar);
        let out_max_tile: Tile<f32, { [1] }> = out_max_scalar.reshape(const_shape![1]);
        let out_idx_scalar: Tile<i32, { [] }> = scalar_to_tile(idx_scalar);
        let out_idx_i32: Tile<i32, { [1] }> = out_idx_scalar.reshape(const_shape![1]);
        let out_idx_tile: Tile<u32, { [1] }> = bitcast(out_idx_i32);
        block_max.store(out_max_tile);
        block_idx.store(out_idx_tile);
    }

    #[cutile::entry(print_ir=false,
        optimization_hints = (
          sm_120 = (occupancy=1, max_divisibility=16,),
        ))]
    fn embedding_f16<const D: i32, const BLOCK_SIZE: i32>(
        token_ids: &Tensor<u32, { [-1] }>,
        table: &Tensor<f16, { [-1, D] }>,
        out: &mut Tensor<f16, { [1, D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        // partition the token_ids with shape [1], we will compute each token id in a separate tile.
        let ids_part = token_ids.partition(const_shape![1]);
        let token_tile: Tile<u32, { [1] }> = ids_part.load([row]);
        // u32 to i32, because the table is a 2D tensor with shape [-1, D], and the token_idx is an index in the table.
        let token_idx_tile: Tile<i32, { [1] }> = bitcast(token_tile);
        let token_idx: i32 = tile_to_scalar(token_idx_tile.reshape(const_shape![]));

        let emb_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            table.partition(const_shape![1, BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(const_shape![1, BLOCK_SIZE]) };
        for j in 0i32..(D / BLOCK_SIZE) {
            let emb: Tile<f16, { [1, BLOCK_SIZE] }> = emb_part.load([token_idx, j]);
            unsafe { out_part.store(emb, [0i32, j]) };
        }
    }

    /*
    It is a wrong implementation, because the table is not partitioned.
    Then the memory could be accessed out of bounds.
    */
    #[cutile::entry(print_ir=false,
        optimization_hints = (
          sm_120 = (occupancy=1, max_divisibility=16,),
        ))]
    fn embedding_f16_2<const D: i32>(
        token_ids: &Tensor<u32, { [-1] }>,
        table: &Tensor<f16, { [-1, D] }>,
        out: &mut Tensor<f16, { [1, D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let ids_part = token_ids.partition(const_shape![1]);
        let token_tile: Tile<u32, { [1] }> = ids_part.load([row]);
        let token_idx_tile: Tile<i32, { [1] }> = bitcast(token_tile);
        let token_idx: i32 = tile_to_scalar(token_idx_tile.reshape(const_shape![]));

        let table_tile = table.partition(const_shape![1, D]).load([token_idx, 0i32]);
        out.store(table_tile);
    }

    /*
        embedding_f16（1 CTA 1 token）更好，v3 基本是负优化：
    - 总工作量一样：都是 seq_len × D/BLOCK_SIZE 次 [1,BLOCK_SIZE] gather，max_divisibility=16 的向量化也一样，没有带宽节省。不同 token 查的是 table 不同行，互相无复用，塞进一个 CTA 也产生不了局部性。
    - 并行度 v1 赢：grid = seq_len vs seq_len/N。prefill 时 v1 有 N 倍多的独立 block 去填 SM；
    v3 把 N 个无关联 token 串行塞进一个 CTA（for i 循环内 load→store 有数据依赖，不能重叠），只是拉长了单个 CTA 的寿命，还制造了 tail 问题（seq_len % N != 0 就要特殊处理，v1 任意长度都行）。
    - v3 唯一理论赢点是单 CTA 工作太少、launch 开销占比高时，用 N 聚批摊薄开销。但 D=4096/BLOCK=128 时单 CTA 已有 32 次迭代，足够掩盖延迟，多这一层循环收益可忽略。
    结论：保留 v1，要调优调 BLOCK_SIZE（对齐 16 的 64/128），而不是按 N 聚批。
     */
    #[cutile::entry(print_ir=false,
        optimization_hints = (
          sm_120 = (occupancy=1, max_divisibility=16,),
        ))]
    fn embedding_f16_3<const N: i32, const D: i32, const BLOCK_SIZE: i32>(
        token_ids: &Tensor<u32, { [-1] }>,
        table: &Tensor<f16, { [-1, D] }>,
        out: &mut Tensor<f16, { [N, D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        // One CTA handles N tokens: token_ids partitioned by [N], out partitioned by [N, D].
        let ids_part = token_ids.partition(const_shape![1]);
        let emb_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            table.partition(const_shape![1, BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(const_shape![1, BLOCK_SIZE]) };
        for i in 0i32..N {
            let token_tile: Tile<u32, { [1] }> = ids_part.load([row * N + i]);
            // u32 to i32, because the table is a 2D tensor with shape [-1, D], and the token_idx is an index in the table.
            let token_idx_tile: Tile<i32, { [1] }> = bitcast(token_tile);
            let token_idx: i32 = tile_to_scalar(token_idx_tile.reshape(const_shape![]));

            for j in 0i32..(D / BLOCK_SIZE) {
                let emb: Tile<f16, { [1, BLOCK_SIZE] }> = emb_part.load([token_idx, j]);
                unsafe { out_part.store(emb, [i, j]) };
            }
        }
    }
}
