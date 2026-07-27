#



nvprof 或 Nsight Compute

第二周：三大经典算子通关（手感与算力压榨）
大模型推理本质上就是矩阵乘法（GEMM）和激活函数/Softmax（Element-wise & Reduction）的组合。这周死磕这三个算子的优化。

核心知识点：

Memory Coalescing（合并访存）：怎么让线程连续访问内存以吃满带宽。

Bank Conflict（存储体冲突）：Shared Memory 的致命陷阱与消除方法。

Reduction（归约算法）的并行优化（树状优化、Warp Shuffle 乱序指令）。

动手实践（必须亲手写）：

基础版 GEMM：用最朴素的全局内存实现。

分块版 GEMM（Tiled Matrix Multiplication）：利用 Shared Memory 减少对全局内存的访问（这一步迈过去，你就懂了 CUDA 的一半）。

Parallel Reduction：实现一个高性能的数组求和或 Max 算子（为后面的 Softmax 打基础）。

第三周：大模型推理核心——FlashAttention 与 Triton（降维打击）
大模型推理最大的痛点在 Attention 的 KV Cache 访存。懂 FlashAttention 原理是现在的标配。同时引入 Triton，用现代手段加速。

核心知识点：

FlashAttention 原理：不要求你手写完整的 FlashAttention CUDA Kernel（一个月不够），但必须搞懂它如何通过 Online Softmax 和 Tiling 避免写回 Global Memory。

Triton 编程：学习 OpenAI 的 Triton 语言。它的 Block-based 编程思维比原生 CUDA 更贴近大模型算子。

动手实践：

用 Triton 实现一个简单的 Matrix Multiplication 和 LayerNorm。

对比 Triton 算子和 PyTorch 原生算子的 Speedup，体会算子融合（Kernel Fusion）的威力。

第四周：主流推理框架源码拆解与量化实战（贴近就业）
最后一周回到业务层，看看行业大厂目前在用什么，把底层知识和上层框架串联起来。

核心知识点：

vLLM 核心机制：深入理解 PagedAttention 原理（如何解决内存碎片）。

TensorRT 工作流：模型导出的静态图优化、算子折叠。

量化算法（Quantization）：INT8/FP8/INT4，理解 AWQ、GPTQ 在底层是如何通过 Weight-only 量化减少访存的。

动手实践：

阅读 vLLM 中 PagedAttention 的 Kernel 源码（或精简版实现），尝试看懂它在 CUDA 层面的线程划分。

使用 TensorRT 或者是 vLLM 部署一个 Llama 3 级别的模型，做一次 Benchmark（吞吐量与延迟测试）。