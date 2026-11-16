# Triton

## Grid in triton
它俩其实是同一个东西，只是省略了尾部的 1
grid 这个 tuple 的含义是「每个维度上启动多少个 CTA」，长度 = 你实际用到的维度数。(64,) 等价于 CUDA 的：
kernel<<<dim3(64, 1, 1), blockDim>>>(...)

dim3 永远有 x/y/z 三个分量，未指定的默认就是 1；Triton 让你把这几个 1 省掉。实测（打印 tl.num_programs(0/1/2)）：
grid 写法	num_programs	启动的 CTA 数
(64,)	[64, 1, 1]	64
(64, 1)	[64, 1, 1]	64
(64, 1, 1)	[64, 1, 1]	64
(8, 8)	[8, 8, 1]	64
(2, 3, 4)	[2, 3, 4]	24

所以三种一维写法完全等价。softmax.py 里那句 kernel[(num_programs, 1, 1)] 就是把 1 写全的啰嗦版，没有别的含义。
源码里就是这么补的
triton/runtime/jit.py 的 run()：
if callable(grid):
    grid = grid(bound_args)          # autotune 时在这里变成 (32,)
grid_size = len(grid)
grid_0 = grid[0]
grid_1 = grid[1] if grid_size > 1 else 1     # ← 缺省补 1
grid_2 = grid[2] if grid_size > 2 else 1     # ← 缺省补 1
kernel.run(grid_0, grid_1, grid_2, stream, ...)
最后传给驱动的一定是三个数，和 CUDA 的 dim3 一一对应。


Triton 里通常写 1D 的原因
1. 一维线性编号已经够用：block_id = tl.program_id(0) 就是全局第几块，一个问题维度用 cdiv(n, BLOCK) 一把算完，不需要额外维度。
2. 维度上限不同：gridDim.x 上限 2³¹−1，而 gridDim.y/z 只有 65535。大张量如果摊到 y/z 上，很容易超限（n=2**28 配 BLOCK=1024 就已经 262144 块 > 65535）。
3. 少一层除法/取模：2D grid 里想拿线性编号要 pid0 + pid1 * num_programs(0)，多一条整数运算。
反过来，用 2D/3D 的唯一好处是：坐标直接对应（比如 (tile_x, tile_y) 或 (seq_block, batch)），kernel 里不用写整除取模。它不会改变调度方式 —— 上一轮已经验证过，CTA 落到哪个 SM 和维度含义无关。

两个必须注意的坑
1. grid 的维度和你读的 program_id 必须配套。你上一版写 grid=(1, 64) 却只读 tl.program_id(0)，64 个 CTA 的 pid0 全是 0，就是这个原因错的。要么用 1D grid，要么把 pid1 编进地址。
2. 多传一个维度不会报错，会被静默丢掉（3.9 实测）：
grid=(2, 3, 4, 5)  →  num_programs=[2, 3, 4]，只启动 24 个 CTA   # 第 4 个元素被忽略
所以要自己保证 len(grid) ≤ 3。
还有一处和 CUDA 不一样

CUDA 的 <<<grid, block>>> 有两半，Triton 的 kernel[grid] 只对应 grid 那一半；block 那一半由 num_warps 决定（num_warps × 32 个线程，CDNA 上是 × 64）：
add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024, num_warps=4)   # 128 线程/CTA
#                  ↑ 只有这一半                      ↑ 另一半
换句话说：grid=(64,) + num_warps=4 ≈ dim3 grid(64); dim3 block(128); kernel<<<grid, block>>>(...)。


## about threadId.X
在 OpenAI Triton 的编程模型中，你无法（也不需要）直接取到类似 CUDA 中的 threadIdx.x。这是因为 Triton 采用的是块级（Block-level / Tile-level）编程模型，而不是传统 CUDA 的 SIMT（单指令多线程）线程级编程模型。具体原因、替代方案以及底层的映射关系可以从以下几个方面来理解：

一、 为什么 Triton 没有 threadIdx.x？抽象级别的提升：在 Triton 中，你编写的逻辑是以整个 Tile（数据块） 为单位的（例如一次性加载 $128 \times 128$ 的矩阵块），而不是去操纵某一个单独的硬件线程（Thread）。编译器自动托管：寄存器分配、线程束（Warp）调度、以及具体把数据分发到哪个物理线程的寄存器上，全部由 Triton 编译器（MLIR 栈）在后台自动完成。因此，暴露底层绝对线程索引（如 threadIdx.x）会破坏这种高阶抽象。

二、 如果需要“线程维度的操作”，在 Triton 里怎么做？虽然没有 threadIdx.x，但 Triton 提供了高阶的原语来让你在块内部表达类似“并行计算”或“相对位置”的逻辑：1. 生成块内的相对坐标/索引：tl.arange如果你需要知道当前 Tile 内部某个元素相对于 Tile 起始位置的偏移量，可以使用 tl.arange。这在逻辑上类似于在 Block 内部生成了一张坐标网格：Pythonimport triton.language as tl

### 生成一个大小为 BLOCK_SIZE 的一维索引向量 (0, 1, 2, ..., BLOCK_SIZE-1)
offsets = tl.arange(0, BLOCK_SIZE)
如果你在写 2D Tile，也可以组合生成二维坐标：Pythonoffs_m = tl.arange(0, BLOCK_M)
offs_n = tl.arange(0, BLOCK_N)

### 通过广播（Broadcasting）形成二维网格坐标
2. 获取程序实例 ID：tl.program_id类似于 CUDA 中的 blockIdx.x / blockIdx.y，Triton 提供了获取当前正在执行的 Grid 块索引的原语：Pythonpid_m = tl.program_id(axis=0)  # 获取当前块在第 0 维的逻辑 ID
pid_n = tl.program_id(axis=1)  # 获取当前块在第 1 维的逻辑 ID
通过 pid 乘以 BLOCK_SIZE，再加上 tl.arange 生成的局部偏移，就能计算出每个元素在全局显存（HBM）中的绝对地址。

