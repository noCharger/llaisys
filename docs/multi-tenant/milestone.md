M1: 在不改变内存布局（保持 Contiguous Tensor）的前提下，改造 Scheduler 和 ModelRunner ，使其支持在一个 Step 中同时处理多个请求（有的在 Prefill，有的在 Decode）

Continuous Batching (连续批处理 / Orca 调度) ：

- 定义 ：这是一种 动态调度策略 。它也调用底层的 forward ，但它是 Iteration-level（迭代级） 的。
- 机制 ：
  1. 调度器维护一个 Running Queue 。
  2. 在每一个 Token 生成步骤（Iteration）结束时，它会检查：
     - 有没有请求已经生成完了？有 -> 立即移除 (Eject) ，释放显存槽位。
     - 有没有新请求在排队？有 -> 立即插入 (Inject) 到空出的槽位中。
  3. 关键点 ：它让 Batch 中的请求 可以随时加入、随时退出 。
- 底层要求 ：为了实现 Continuous Batching，底层的 forward 必须支持 不规则的 Batch （即：Request A 是第 1 个 token，Request B 是第 50 个 token）。这通常需要 Attention Mask 和 Positional Encoding 的特殊处理。

M2: Gateway 适配新的缓存亲和性路由（将相同 Prompt 的请求路由到同一实例以命中 Cache）。

在 M1 Milestone (Continuous Batching + Contiguous Memory) 的架构下，算子层面（Operator Level）确实需要做一些关键的改造，主要是为了支持 不规则 Batch (Irregular Batch / Ragged Batch) 。

### 需要实现的算子改造
我会在文档的 2.5 Execution Engine 部分补充以下细节：

1. Rotary Positional Embeddings (RoPE) for Ragged Batch :
   
   - 现状 ：通常的 RoPE 实现假设 Batch 中的所有序列长度对齐，或者位置索引是连续的 0...N 。
   - 需求 ：在 Continuous Batching 中，Batch 里包含了来自不同请求的 Token，它们处于不同的生成阶段。
     - Request A: 生成第 10 个 Token (Pos = 9)
     - Request B: 生成第 100 个 Token (Pos = 99)
   - 实现 ：需要传入一个额外的 position_ids Tensor [Batch_Size] ，Kernel 内部根据每个 Token 对应的 position_id 计算旋转角度，而不是简单地用 threadIdx.x 。
2. Masked Self-Attention for Ragged Batch (Variable Length) :
   
   - 现状 ：标准的 FlashAttention 或 SDPA (Scaled Dot-Product Attention) 支持 cu_seqlens (Cumulative Sequence Lengths) 参数来处理变长序列。
   - 需求 ：确保集成的 Attention 算子（无论是调库还是手写）正确处理了 Batch Packing 。
     - 输入不再是 [Batch, Max_Len, Head, Dim] 。
     - 输入通常是 [Total_Tokens, Head, Dim] (Flattened)。
   - 实现 ：你需要构建 cu_seqlens_q (Query 的累积长度) 和 cu_seqlens_k (Key 的累积长度)，传给 FlashAttention。
3. RMSNorm & Linear (GEMM) - No Change Needed :
   
   - 这两个算子是 Element-wise 或 Token-independent 的。只要把输入 Flatten 成 [Total_Tokens, Hidden_Dim] ，它们就能直接跑，完全不需要感知属于哪个请求。这是 Transformer 的一个巨大优势。