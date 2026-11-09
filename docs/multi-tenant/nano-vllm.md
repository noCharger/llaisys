System Architecture
Component Hierarchy
The system is organized into four logical layers that process inference requests from user input to generated tokens:


















Layer Responsibilities:

User Interface Layer: Provides the LLM class as the main entry point, accepting prompts and SamplingParams for generation control
Inference Engine Layer: LLMEngine orchestrates the request lifecycle, coordinating tokenization, scheduling, and execution
Memory Management Layer: BlockManager implements paged KV cache allocation, while Sequence tracks per-request state
Model Execution Layer: ModelRunner executes the neural network, managing distributed computation and CUDA graph optimization
Sources: 
README.md
36-44
 
pyproject.toml
1-28

Request Processing Pipeline
Each inference request flows through the system following this execution path:

Execution Phases:

Tokenization: LLMEngine converts text prompts to token IDs using HuggingFace AutoTokenizer
Scheduling: Scheduler determines which sequences to process (prefill vs. decode) based on resource availability
Memory Allocation: BlockManager allocates or extends KV cache blocks for selected sequences
Model Execution: ModelRunner performs forward pass and samples next tokens
State Update: Scheduler updates sequence states, detecting finished generations
Decoding: LLMEngine converts completion tokens back to text
Sources: 
README.md
36-44

Core Optimization Techniques
Nano-vLLM implements four major optimizations that enable high-throughput inference:

1. Prefix Caching (Automatic KV Cache Reuse)













The BlockManager implements content-addressable caching by hashing token sequences (using xxhash) and maintaining a hash_to_block_id mapping. When multiple sequences share identical prefixes, they reference the same physical KV cache blocks, reducing memory usage and enabling reuse of previous computation.

Implementation: Block allocation checks hash_to_block_id before creating new blocks. Reference counting (ref_count) tracks block usage, allowing automatic deallocation when no sequences reference a block.

2. Tensor Parallelism (Multi-GPU Scaling)
The system shards model weights across multiple GPUs to enable inference of models larger than single-GPU memory. Sharding occurs along the head dimension for attention and hidden dimension for MLP layers:

Layer Type	Sharding Strategy	Communication
VocabParallelEmbedding	Vocabulary split across GPUs	No communication (duplicated inputs)
QKVParallelLinear	Heads divided by tp_size	No communication (local attention)
ColumnParallelLinear	Output dim divided by tp_size	No communication (activation sharded)
RowParallelLinear	Input dim divided by tp_size	all_reduce to combine partial sums
ParallelLMHead	Vocabulary split across GPUs	gather to rank 0 for sampling
Coordination uses torch.distributed with NCCL backend for efficient GPU-to-GPU communication.

3. CUDA Graphs (Kernel Launch Optimization)
For decode phase with small batches (≤512 sequences), ModelRunner captures the computation graph once and replays it repeatedly, eliminating per-iteration kernel launch overhead:

# Capture phase (first decode iteration)
model.forward(inputs) -> graph = torch.cuda.CUDAGraph()
 
# Replay phase (subsequent iterations)
copy inputs to graph_vars
graph.replay()
CUDA graphs reduce CPU overhead by 10-20% for decode operations. Prefill operations and large batches use eager execution for flexibility.

4. Torch Compilation (Kernel Fusion)
Selected components use torch.compile() with fullgraph=True mode to fuse operations:

RMSNorm with residual: Fuses normalization and addition into single kernel
Small linear layers: Reduces memory traffic for low-dimensional operations
Compilation happens lazily during the first forward pass.

Sources: 
README.md
11-17

Key Components Overview
The table below maps high-level functionality to specific implementation classes:

Functionality	Primary Class	File Path	Purpose
User API	LLM	nanovllm/__init__.py	Main interface for text generation
Request Orchestration	LLMEngine	nanovllm/engine.py	Coordinates tokenization, scheduling, execution
Batch Scheduling	Scheduler	nanovllm/scheduler.py	Manages sequence lifecycle and batching
Memory Management	BlockManager	nanovllm/block_manager.py	Allocates/frees KV cache blocks
Sequence Tracking	Sequence	nanovllm/sequence.py	Tracks per-request state and tokens
Model Execution	ModelRunner	nanovllm/model_runner.py	Executes model with optimizations
Neural Network	Qwen3ForCausalLM	nanovllm/models/qwen3.py	Transformer model implementation
Token Sampling	Sampler	nanovllm/sampler.py	Selects next tokens from logits
Configuration	Config	nanovllm/config.py	System parameters and validation
For detailed documentation of each component, see Core Engine Components and Model Architecture.