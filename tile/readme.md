CUDA_TOOLKIT_PATH="/usr/local/cuda-13.3" PATH="/usr/local/cuda-13.3/bin":$PATH cargo run



# cutile

## 
cutile 抓住三层，职责别混：
1. 心智模型
- Tensor = 全局显存，维度可动态（[-1]）。
- Partition = 网格视图。host 端 .partition([...]) 决定 grid（几个 CTA、每个 CTA 拥哪块）；kernel 内 .partition(...) 决定取数视角。
- Tile = 片上（寄存器/shared），形状必须编译时常量。Tile 一大就爆寄存器/occupancy，这是所有维度设计的根本约束。
2. 设计顺序：从输出倒推
1. 先定 out 的 CTA 映射：1 CTA 写 1 个输出块（add: [B]、gemm: [BM,BN]、embedding: [1,D]）。调用前按这个粒度 partition，pid 只用来选输入，不用来选输出（z.store 写自己那块）。
2. 再定 CTA 内部循环：并行维（M/N/seq）→ grid（pid.0/pid.1），规约维（K）→ CTA 内 for + 累加器 Tile（gemm 的 K/BK 循环、sum 的 reduce_sum 循环就是标准形）。
3. 最后定块大小：模型维度（D/K）和硬件块（BLOCK_SIZE/BK）解耦，D/BLOCK_SIZE 个小块循环搬运，保证 max_divisibility=8/16 对齐。别整行一个 Tile（embedding_f16_2 的教训）。
3. 几个范式（都在你仓库里）
- 逐元素：load_tile_like + store（add_1d/silu），1 CTA 1 块，无循环。
- 规约：循环 load → reduce_sum/max → acc（sum_1d），needs 两遍则拆 stage1/stage2。
- GEMM：2D grid + K 循环 + mma + convert_tile。
- gather：pid → 标量 index → 分块 copy（embedding），1 CTA 写多块时才用 partition_mut + store(tile,[i,j]) + unsafe，否则 store 就够。
- 越界：要么要求整除，要么像 argmax 那样 valid mask + -inf 显式处理。
4. 检查清单
out 不能是 [-1]；host 切分粒度必须等于 kernel 假设的 CTA 粒度（按行还是按块）；PartitionMut 只给输出；BLOCK_SIZE 取 32/64/128 且整除 D；先小尺寸测对（32）再放大到 4096 验证资源。

