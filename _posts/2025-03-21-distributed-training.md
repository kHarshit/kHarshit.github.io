---
layout: post
title: "Distributed Training: How to train Large Language Models (LLM)"
date: 2025-03-21
categories: [LLM, Generative AI, Deep Learning]
permalink: /blog/distributed-training/
excerpt: "Comprehensive guide to distributed training for LLMs covering data parallelism, model parallelism, tensor parallelism, ZeRO optimizer, FSDP, 3D parallelism, DeepSpeed with interactive visualization, code examples."
mathjax: true

---

Training large language models requires large amounts of GPU memory, far beyond what a single GPU can provide. This post explores key distributed training strategies that make it possible to train models with billions of parameters across hundreds of GPUs.

## 1. Background

For a 10B parameter LLM, it requires ~176 GiB of GPU memory (FP32: 4 bytes/param, FP16: 2 bytes/param) for mixed precision training. 

| Component | Precision | Explanation | Memory |
|-----------|:------:|:------:|:------:|
| Parameters / weights | bf16 | 10B * 2 bytes | 20 GB |
| Gradients            | bf16 | 10B * 2 bytes | 20 GB|
| Optimizer states     | fp32 | AdamW: momentum + variance <br> 2 * (10B * 2 bytes) | 80 GB |
| FP32 master weights  | fp32 | Used in mixed precision training <br> 10B * 4 bytes | 40 GB |
| Activations          | bf16 | Dependent on batch size & sequence length | ~20 GB |
| Temporary buffer     | mixed | Attention, matmul, CUDA workspace (mixed) | ~10 GB |
| Total                | | | 190 GB (~176 GiB) |
{:.mbtablestyle}

{% include interactive/distributed-training-memory_calc.html %}

The A100 GPU has 80GB memory. Thus, for a 10B model, you'd need 3 A100s just to hold the parameters + optimizer states. Distributed training is essential to train Large Language Models. 

## 2. Scaling 

There are two fundamental scaling approaches:

<div class="mbgrid mbgrid-2" markdown="1">
<div class="mbcard" markdown="1">
**Horizontal Scaling (Scale Out)**
- Add more machines/instances to distribute workload across smaller resources
- Easier to scale dynamically
- Requires more complex management
</div>
<div class="mbcard" markdown="1">
**Vertical Scaling (Scale Up)**
- Increase capacity of existing machine (more CPU, RAM, storage)
- Easier to manage
- Hardware upgrades can require downtime
</div>
</div>

In distributed training, we're mainly working with horizontal scaling since machine specification is fixed e.g. `p5.48xlarge` AWS instance consists of 8xA100 GPUs with fixed memory and CPUs. And, also, a machine can only be scaled up to a point so we need to figure out to split our data or model on multiple GPUs machines. Distributed training is all about how to do that.

## 3. Communication Primitives

Before diving into parallelism strategies, it helps to understand the underlying communication operations.

### Point-to-Point Communication

Direct transfer of data between two specific processes (send/receive).

{% include img.html src="/img/blog/distributed-training/communication_primitives_point_to_point.jpg" caption="Communication Primitives: Point to Point" %}

### Collective Communication

Operations involving all processes in a group simultaneously.

{% include img.html src="/img/blog/distributed-training/communication_primitives_collective.jpg" caption="Communication Primitives: Collective" %}

**AllReduce** is the key operation for synchronizing gradients across GPUs at the end of each training iteration e.g., average the gradients from different nodes, then use the averaged value to update weights. Used for data parallelism.

Steps in AllReduce-based data parallel training:
1. **Step 0:** Data is fetched from store to all nodes participating in distributed training.
2. **Step 1:** During the forward pass, each model copy does a forward pass with its batch of data.
3. **Step 2:** A backward pass is performed to compute gradients. The gradient is **NOT** used to update weights yet.
4. **Step 3:** An AllReduce operation runs across all processes (average gradients and then broadcast).
5. **Step 4:** The final all-reduced gradients are used to update each model replica.

**AllGather** is another key operation used in sharded data parallel training (e.g., FSDP, DeepSpeed ZeRO).

The **Divide-and-Conquer** approach is followed during Broadcast and Reduce operations.

**Ring AllReduce** arranges GPUs in a logical ring, where each node communicates only with its two immediate neighbors. It requires `2(N-1)` communication steps total.

<div class="mbsteps" markdown="1">
<div class="mbstep" markdown="1">
**Phase 1: Reduce-Scatter (`N-1` steps)**
Each GPU splits its vector (e.g., gradients) into `N` chunks. Chunks circulate clockwise around the ring: each GPU receives a chunk, adds its own corresponding chunk, and forwards the partial sum. After `N-1` steps, each GPU holds exactly one fully-reduced chunk.
</div>
<div class="mbstep" markdown="1">
**Phase 2: All-Gather (`N-1` steps)**
The reduced chunks circulate around the ring again. Each GPU sends the reduced chunk it owns to its neighbor, receives the incoming chunk, stores it, and forwards it. After `N-1` steps, every GPU has all reduced chunks.
</div>
</div>

**Example:** Try the interactive visualization below to see the full 6-step AllReduce algorithm.

{% include interactive/distributed-training-allreduce.html %}

## 4. Data Parallelism

Used when the model can fit in a single GPU. Each device (worker) holds a full copy of the model, but processes a different batch of training data. This way, data parallelism can scale up the training.

### Data Parallelism Steps

{% include img.html src="/img/blog/distributed-training/data_parallelism.jpg" width="70%" caption="Data Parallelism" %}

1. **Broadcast:** Model weights are initialized on one GPU worker and broadcast to all other nodes.
2. **Forward pass:** Each GPU worker has the same model (weights $$W$$) but processes different mini-batches $$X_i$$.
3. **Backward pass:** Each worker computes a weight gradient $$dW_i$$ for its portion of weight parameters on local mini-batch.
4. **Gradient synchronization:** The gradients from each worker are averaged across all workers via `AllReduce` operation. Communication and computation can overlap with AllReduce gradients for layer $$k$$ while computing gradients for layer $$k-1$$.
6. **Update:** Each worker updates its local model parameters with the average gradients $$\overline{dW}$$ using its own optimizer. 
    - $$\overline{dW} = (dW_0 + dW_1 + dW_3) / 3$$, then
    - $$W_1 = W - lr * \overline{dW}$$. 
    - After the update, all workers have the same updated model weigths.
7. **Repeat:** Go back to step 2 for next mini-batch.

The total **global batch size** is defined as the total records sent to all GPUs per iteration = `(num of GPUs) × (per-replica batch size)`.

### Parameter-server approach

An alternative approach for gradient synchronization is to use a separate server that stores parameters. In this setup, workers send gradients to parameter servers, the servers aggegrate the gradients and redistribute the model parameters. 

Workers use a push-and-pull pattern:

* push gradients -> parameter server
* pull updated parameters <- parameter server

It can be synchronous (end of each training step) or asynchronous (replicas push/pull independently).

### Key Points

* Data parallelism improve the overall throughput, but doesn't reduce model memory per GPU.
* Each GPU worker processes roughly 1/N of global batch. However, each worker still stores the full model and performs full optimizer update for all parameters. Techniques like optimizer sharding, ZeRO, or FSDP can reduce this redundancy.


### PyTorch Distributed Data Parallel (DDP)

The following code implements data parallelism with gradient accumulation:
{% highlight python %}
def train():
    if global_rank == 0:
        initialize_services () # W&B, etc.
    data_loader = DataLoader(train_dataset, shuffle=False, sampler=DistributedSampler(train_dataset, shuffle=True))
    model = MyModel()
    if os path.exists('latest_checkpoint.pth'): # Load latest checkpoint
        # Also load optimizer state and other variables needed to restore the training state
        model. load_state_dict(torch.load('latest_checkpoint.pth'))
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), Ir=10e-4, eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range (num_epochs) :
        for data, labels in data_loader:
            if (step_number + 1) % 100 != 0 and not last_step: # Accumulate gradients for 100 steps
                with model.no_sync(): # Disable gradient synchronization
                    loss = loss_tn(model(data), labels) # Forward step
                    loss.backward() # Backward step + gradient ACCUMULATION
            else:
                loss = loss_fn(model(data), labels) # Forward step
                loss.backward() # Backward step + gradient SYNCHRONIZATION
                optimizer.step() # Update weights
                optimizer.zero_grad() # Reset gradients to zero
            if global_rank == 0:
                collect_statistics () # W&B, etc.
        if global_rank == 0: # Only save on rank o
            # Also save the optimizer state and other variables needed to restore the training state
            torch.save(model.state_dict(), "latest_checkpoint.pth')

if _name_ == '_main_':
    local_rank = int(os.environ['LOCAL_RANK' ])
    global_rank = int(os. environ ['RANK'])
    init_process_group (backend='nccl')
    torch.cuda.set_device(local_rank) # Set the device to local rank
    train()
    destroy_process_group()

# Run on all machines:
torchrun \
  --nnodes=NUM_NODES \
  --nproc-per-node=TRAINERS_PER_NODE \  # GPUs per node
  --max-restarts=NUM_ALLOWED_FAILURES \
  --rdzv-id=JOB_ID \
  --rdzv-backend=c10d \
  --rdzv-endpoint=HOST_NODE_ADDR \
  YOUR_TRAINING_SCRIPT.py [--arg1 ...]
{% endhighlight %}


## 5. Model Parallelism

Used when the model is too big to fit in a single GPU.

### Pipeline Parallelism (Inter-layer)

Pipeline parallelism partitions the model's layers across multiple GPUs. The training mini-batch is split into micro-batches that flow through pipeline. The forward and backward computation of micro-batches are overlapped to reduce device idle time.

{% include img.html src="/img/blog/distributed-training/pipeline_parallelism.jpg" width="60%" caption="Pipeline Parallelism" %}

The pipeline parallelism on 4 stages on 4 GPU devices involves following steps.

1. **Partition model** into 4 sequential stages and place each stage on a different device.
2. **Split global mini-batch** into M micro-batches.
3. **Forward Pass:**
    - **Pipeline Fill:** Stage 0 on GPU 0 starts with micro-batch 0 and sends activations to Stage 1 on GPU 1. Each next stage starts when it receives activations.
    - **Steady State:** All stages are busy. While Stage i works on micro-batch `k`, Stage `i-1` can work on MB `k+1`.
    - **Drain:** The last stage finishes remaining micro-batches and computes the loss.
6. **Backward Pass (Drain → Fill):** Gradients flow backward from Stage 3 to Stage 0 in the reverse order.
7. **Update Parameters:** Each stage updates only its own parameters using the gradients it computed. Gradients are applied
synchronously at the end.
8. Repeat for the next global mini-batch.

{% include img.html src="/img/blog/distributed-training/pipeline_parallelism_bubble.jpg" width="70%" caption="Pipeline Parallelism (source: GPipe paper)" %}

At the beginning, later stages are idle while the first micro-batch moves through pipeline. At the end, earlier stages become idle while the last backward computations finish. This idle time is called **pipeline bubble**. Increasing the number of micro-batches reduces the relative bubble overhead, but using too many can also increase scheduling complexity.

{% include interactive/distributed-training-pipeline_viz.html %}

#### Interleaved Layers

In interleaved pipeline parallelism, non-contiguous layers (e.g., layer 1 and layer 4) are assigned to GPU workers instead of consecutive layers. This reduces worker idle time but increases communication overhead (worker communicates after every layer instead of every 2 layers). It's can be complicated if model has skip connections, attention patterns that cross workers.

{% include img.html src="/img/blog/distributed-training/pipeline_parallelism_interleaved.jpg" width="70%" caption="Pipeline Parallelism: Interleaved" %}

#### 1F1B (One Forward, One Backward) Schedule

In classic data parallsielm, all micro-batches do all forward passes before any backward passes begin. In **1F1B**:
<div class="mbsteps" markdown="1">
<div class="mbstep" markdown="1">
**Warm-up phase**
Workers perform differing numbers of forward passes.
</div>
<div class="mbstep" markdown="1">
**Steady state**
Each worker performs one forward pass followed by one backward pass (unlike classic data parallelism where backward follows forward for all batches).
</div>
<div class="mbstep" markdown="1">
**Drain phase**
Complete backward passes for all remaining in-flight micro-batches.
</div>
</div>

The default non-interleaved 1F1B has a smaller pipeline bubble than GPipe. The interleaved 1F1B (each device assigned multiple chunks) reduces the bubble size further.

{% include img.html src="/img/blog/distributed-training/pipeline_parallelism_1f1b.jpg" width="70%" caption="Pipeline Parallelism: 1F1B (source: 1F1B)" %}

#### Combining Pipeline Parallelism with Data Parallelism 

In this example, we split the model into 2 pipeline stages (Stage 0 and Stage 1). Each stage is replicated across 4 GPUs for data parallelism. Thus, `total GPUs = 2 (pipeline) * 4 (data) = 8`.

{% include img.html src="/img/blog/distributed-training/pipeline_data_parallelism2.jpg" width="80%" caption="8 GPUs with 2-way pipeline parallelism and 4-way data parallelism" %}

Here, the data parallel replicas are as of follows.

{% highlight python %}
Pipeline replica Group 0: 
GPU 0: Stage 0
GPU 4: Stage 1

Pipeline replica Group 1: 
GPU 1: Stage 0
GPU 5: Stage 1

etc.
{% endhighlight %}

Pipeline Parallelism Steps:

1. Split global mini-batch into M micro-batches.
2. Each data-parallel in Stage 0 runs forward pass for its micro-batches.
3. Activations are sent to Stage 1 replias; Stage 1 runs forward pass.
4. After last stage produces outputs, backward pass flows from Stage 1 to Stage 0.
5. Gradients are synchronized across data-parallel replicas within each stage using AllReduce.
6. Optimizer updates are applied (per stage or globally, depending on setup).
7. Repeat for next global mini-batch.

{% include img.html src="/img/blog/distributed-training/pipeline_data_parallelism3.jpg" width="75%" %}

{% include img.html src="/img/blog/distributed-training/pipeline_data_parallelism.jpg" width="50%" caption="Combining Pipeline and Data Parallelism" %}

### Tensor Parallelism (Intra-layer)

Tensor parallelism split the individual layer weights and computation across multiple GPUs unlike pipeline parallelism (which keeps individual weights intact but partitions layers). It's required when a single parameter consumes most GPU memory, or for extremely large models like GPT.

There are two ways to split the weight matrix W.

<div class="mbgrid mbgrid-2" markdown="1">
<div class="mbcard" markdown="1">
**Column-wise Partitioning** (by output dimension)
No communication needed until a later layer requires the full output (then AllGather).
</div>
<div class="mbcard" markdown="1">
**Row-wise Partitioning** (by input dimension)
Partial outputs are summed with AllReduce to get the full output.
</div>
</div>

{% include img.html src="/img/blog/distributed-training/tensor_parallelism_partitioning.jpg" width="70%" caption="Tensor Parallelism: Column and Row Partitioning" %}

#### Transformer MLP

A Transformer MLP is usually. `Y = GELU(XA); Z = YB`.

In Megatron-LM tensor parallelism, the first GEMM weight matrix `A` is column-partitioned $$A = [A1, A2]$$ so that GeLU nonlinearity can be applied independently to each partitioned GEMM output:

$$[Y_1, Y_2] = [\text{GeLU}(XA_1),\ \text{GeLU}(XA_2)]$$

*If we had split A into rows $$\begin{bmatrix}A1 \\ A2 \end{bmatrix}$$, a sync point would have been needed since `GeLU(X1A1 + X2A2) ≠ GeLU(X1A1) + GeLU(X2A2)`.*

The second GEMM matrix `B` is row-partitioned $$\begin{bmatrix}B1 \\ B2 \end{bmatrix}$$

$$ Z_1 = [Y_1 B_1]; Z_2 = [Y_2 B_2] $$

$$Z = \text{AllReduce}(Z_1,\ Z_2)$$

{% include img.html src="/img/blog/distributed-training/tensor_parallelism_mlp.jpg" width="70%" caption="Tensor Parallelism: Column + Row Partitioning of MLP (source: Megatron-LM paper)" %}

The advantage of partitioning the first MLP GEMM column-wise and the second MLP GEMM row-wise is that no communication is needed in-between until end of MLP blocks. An AllReduce is only needed after row-parallelism.

> Note: Row-wise partitioning in the forward pass becomes column-wise partitioning in the backward pass and vice versa.

#### Multi-Head Attention (MHA)

MHA blocks are natural fit for tensor parallelism due to attention heads being mostly indpendent before final output projection. We can divide Q, K, V weight matrices by columns and the output linear layer by rows. This introduces two AllReduce operations per layer in both forward and backward passes.

{% include img.html src="/img/blog/distributed-training/tensor_parallelism_attention.jpg" width="70%" caption="Tensor Parallelism: Column + Row Partitioning of Multi-Headed Attention (source: Megatron-LM paper)" %}

## 6. Zero Redundancy Optimizer (ZeRO)

ZeRO consists of 3 stages which shards different model states: model parameters (weights), gradients, and optimizer states (e.g., momentum and variance in Adam).

{% include img.html src="/img/blog/distributed-training/deepspeed_zero.jpg" width="70%" caption="ZeRO (source: ZeRO paper)" %}

In the above figure, `Ψ` denotes model size (number of parameters), `K` denotes the memory multiplier of optimizer states, and `Nd` denotes data-parallel degree (#GPUs).

{% include interactive/distributed-training-zero_compare.html %}

### ZeRO Stage 1: Optimizer State Partitioning (P<sub>os</sub>)

Shards optimizer states. Instead of creating per-param states for all parameters on every GPU, each optimizer instance only keeps states for a shard of all model parameters. The optimizer `step()` updates only the parameter shard for which it owns optimizer states and then broadcasts updated parameters to all peers.

### ZeRO Stage 2: Gradient Partitioning (P<sub>os+g</sub>)

Shards both optimizer states and gradients across workers. Each worker maintains gradients only for its parameter partition. DeepSpeed performs a **ReduceScatter** (not AllReduce) so each worker only receives gradients for its own optimizer state partition.

With ZeRO Stage 1 and 2, the entire model must still fit on 1 GPU.

### ZeRO Stage 3: Parameter Partitioning (P<sub>os+g+p</sub>)

Shards all model states (optimizer, gradients, and model parameters). During computation, ZeRO 3 needs its full parameters so it temporarily gathers shards before a layer runs. Its working is quite similar to that of PyTorch FSDP.

Each GPU permanently stores only its own parameter shard, gradient shard, and optimizer shard. It gather the full parameters as needed and free them immediately after computation.

{% include img.html src="/img/blog/distributed-training/zero_example1.jpg" width="70%" %}

{% include img.html src="/img/blog/distributed-training/zero_example2.jpg" width="70%" caption="ZeRO Stage 3 Example" %}

1 **Before Forward:** Each GPU holds only its parameter shards. Before forward pass, AllGather gets all parameters of layer, so every GPU has full parameters temporarily.  
2 **Forward Compute:** Run forward with full parameters.  
3 **After Forward:** Reshard (release) parameters to free memory. {% include img.html src="/img/blog/distributed-training/zero_example3.jpg" width="70%" caption="ZeRO Stage 3 Example: Forward Pass" %}
4 **Backward Compute:** AllGather parameter shards again. Run backward pass to get local gradients.  
5 **After Backward:** ReduceScatter gradients. Gradients are averaged across ranks, each rank keeps only its gradient shard. {% include img.html src="/img/blog/distributed-training/zero_example4.jpg" width="70%" caption="ZeRO Stage 3 Example: Backward Pass" %}
6 **Optimizer Step:** Each rank updates its parameter shard using its optimizer state shard i.e. GPU 0 updates p0 shards, GPU 1 updates p1 shards, GPU 2 updates p2 shards using local optimizer-state shards.   

{% include img.html src="/img/blog/distributed-training/fsdp.jpg" width="70%" caption="ZeRO Stage 3 / FSDP Summary" %}

**ZeRO-Offload / ZeRO-Infinity:**

<div class="mbgrid mbgrid-3" markdown="1">
<div class="mbcard" markdown="1">
**ZeRO-Offload**
Offload optimizer states and gradients to CPU.
</div>
<div class="mbcard" markdown="1">
**ZeRO Offload++**
Offload optimizer and gradient states with better overlap.
</div>
<div class="mbcard" markdown="1">
**ZeRO Infinity**
ZeRO-Offload + offload model weights to CPU/NVMe with better computation and communication overlap.
</div>
</div>

**DeepSpeed Ulysses:** Splits long sequence lengths across workers for sequence parallelism. Useful for long sequence length >10k.

### DeepSpeed Training Setup

You can use Zero via DeepSpeed framework or use PyTorch FSDP for ZeRO Stage 3.

{% highlight python %}
import deepspeed

ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {"lr": 3e-5}
    },
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,                        # ZeRO Stage 3
        "offload_optimizer": {
            "device": "cpu",               # offload optimizer states to CPU
        },
        "offload_param": {
            "device": "cpu",               # offload parameters to CPU
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e6,
    },
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
)

for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
{% endhighlight %}

## 7. PyTorch Fully Sharded Data Parallel (FSDP)

FSDP is a type of data-parallel training, but unlike traditional DDP (which maintains a per-GPU copy of model parameters, gradients, and optimizer states), FSDP shards all of these states across data-parallel workers and can optionally offload sharded parameters to CPU. It is effectively a **mix of data and model parallelism**. *FSDP is PyTorch's equivalent to DeepSpeed ZeRO Stage 3.*

{% include img.html src="/img/blog/distributed-training/fsdp_vs_ddp.jpg" width="70%" caption="DDP vs PyTorch FSDP" %}

**Advantages over DDP:**
- Smaller GPU memory footprint → enables larger models or batch sizes.
- Communication overhead is reduced via overlapping communication and computation.

### How FSDP Works

**FSDP Forward Pass:**
{% highlight python %}
for layer_i in layers:
    all_gather full weights for layer_i   # reconstruct full weights from shards
    forward_pass(layer_i)
    discard full weights for layer_i      # free memory immediately
{% endhighlight %}

**FSDP Backward Pass:**
{% highlight python %}
for layer_i in layers:
    all_gather full weights for layer_i
    backward_pass(layer_i)
    discard full weights for layer_i
    reduce_scatter gradients for layer_i  # average and reshard gradients
{% endhighlight %}

**View as decomposed DDP:** FSDP decomposes DDP's gradient `AllReduce` into a `ReduceScatter` and an `AllGather`:
<div class="mbsteps" markdown="1">
<div class="mbstep" markdown="1">
**Backward pass**
Reduce-scatter gradients: each rank holds a shard of gradients.
</div>
<div class="mbstep" markdown="1">
**Optimizer step**
Each rank updates its parameter shard.
</div>
<div class="mbstep" markdown="1">
**Next forward pass**
AllGather to collect updated parameter shards.
</div>
</div>

{% include img.html src="/img/blog/distributed-training/fsdp_allgather.jpg" width="70%" caption="PyTorch FSDP AllGather" %}

### Wrapping a Model with FSDP

**Auto wrapping** (drop-in DDP replacement):

{% highlight python %}
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    default_auto_wrap_policy,
)
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 4)
        self.layer2 = nn.Linear(4, 16)
        self.layer3 = nn.Linear(16, 4)

# Replace DDP with FSDP:
# model = DistributedDataParallel(MyModel())
fsdp_model = FullyShardedDataParallel(
    MyModel(),
    fsdp_auto_wrap_policy=default_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True),
)
{% endhighlight %}

**Manual wrapping** allows selective application of FSDP to specific parts of the model for complex sharding strategies.


## 8. AWS SageMaker Distributed Training

The SageMaker API can be used for distributed training as follows.

### SageMaker DDP (SMDDP)

{% highlight python %}
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    ...,
    instance_count=2,
    instance_type="ml.p4d.24xlarge",
    # Option 1: mpirun with SMDDP AllReduce OR AllGather
    distribution={"pytorchddp": {"enabled": True}},
    # Option 2: torchrun, activates SMDDP AllGather
    # distribution={"torch_distributed": {"enabled": True}},
    # Option 3: mpirun with smddprun
    # distribution={"smdistributed": {"dataparallel": {"enabled": True}}},
)
{% endhighlight %}

For PyTorch DDP code, simply set the backend to `smddp`:

{% highlight python %}
import torch.distributed as dist
import smdistributed.dataparallel.torch.torch_smddp

dist.init_process_group(backend="smddp")
{% endhighlight %}

SMDDP uses **MPI** (Message Passing Interface) for node communication and **NVIDIA NCCL** for GPU-level communication.

### SageMaker Model Parallelism (SMP)

{% highlight python %}
distribution = {
    "smdistributed": {
        "modelparallel": {
            "enabled": True,
            "parameters": {
                "hybrid_shard_degree": 2,          # degree of sharded data parallelism
                "sm_activation_offloading": True,   # offload activations to CPU
                "activation_loading_horizon": 4,
                "tensor_parallel_degree": 4,
                "expert_parallel_degree": 1,
                "random_seed": 42,
            },
        },
        "mpi": {"enabled": True},
    }
}
{% endhighlight %}

SMP provides Sharded data parallelism, Expert parallelism, Tensor parallelism, Activation checkpointing and offloading, etc funcationalities.


## 9. 3D Parallelism

3D Parallelism combines **Data Parallelism** (or ZeRO/FSDP), **Pipeline Parallelism**, and **Tensor Parallelism** simultaneously. It's used to train most frontier LLMs (GPT-3, LLaMA, etc.) across large cluster of GPUs.

{% include img.html src="/img/blog/distributed-training/3d_parallelism.jpg" caption="3D Parallelism: Data, Pipeline, and Tensor Parallelism" %}

{% include interactive/3d-distributed-training-parallelism_cube.html %}

> Total GPUs = DP degree x PP degree x TP degree

For example, with 2 nodes (16 GPUs), TP=2, PP=4, you'd have DP=N/(TPxPP) = 16/(2×4) = 2 data-parallel groups.

| Dimension | What it splits | Rationale | Communication | Frequency | Network Scope |
|:----------|:--------------|:----------|:-------------|:----------|:--------------|
| **Data Parallel** (DP) | Training data / batch | Increase throughput | AllReduce of gradients (~model params × 2 bytes) | Once per training step | Inter-node (InfiniBand / Ethernet) |
| **Pipeline Parallel** (PP) | Model layers | Fit many layers across GPUs | Send/Receive of activations (~bs × seq_len × hidden) per pipeline stage boundary | Once per micro-batch per stage boundary | Across nodes / groups |
| **Tensor Parallel** (TP) | Matrix operations inside each layer | Split large layer computations; leverage NVLink bandwidth (~600 GB/s) | AllReduce / AllGather / ReduceScatter of activations (~bs × seq_len × hidden) per transformer block | Every forward/backward pass per layer | Intra-node (NVLink / NVSwitch) |
{:.mbtablestyle}

<div class="mbgrid mbgrid-3" style="--mbcard-border: 1.5px solid #d4a0a0; --mbcard-title-color: #e07070" markdown="1">
<div class="mbcard" markdown="1">
**Tensor Parallelism** is the most bandwidth-sensitive, it must run on fast intra-node links (NVLink).
</div>
<div class="mbcard" markdown="1">
**Pipeline Parallelism** reduces memory and the amount of AllReduce data, but introduces the pipeline bubble.
</div>
<div class="mbcard" markdown="1">
**Data Parallelism** provides the most flexibility, you can scale to hundreds of nodes by increasing the DP degree.
</div>
</div>

### ZeRO + 3D Parallelism

In practice, DeepSpeed ZeRO Stage 1 is often used on top of TP + PP instead of full data parallelism. This shards the optimizer states across the data-parallel replicas without adding extra communication. 

{% highlight python %}
# Pseudocode for a 3D-parallel training loop

for batch in dataloader:               # Data Parallel: different data per DP group
    for microbatch in split(batch):     # Pipeline Parallel: micro-batches through stages
        # Tensor Parallel: each layer is split across TP group
        output = tensor_parallel_forward(model, microbatch)
        loss = compute_loss(output)
        # TP backward (AllReduce within node)
        tensor_parallel_backward(loss)
    # PP gradients flow backward across pipeline stages (P2P)
    allreduce_gradients()              # DP: sync gradients across replicas
    optimizer.step()                   # ZeRO: each rank updates its optimizer shard
{% endhighlight %}

## 10. Summary: Parallelism Strategies

| Strategy | Splits | Use Case |
|----------|--------|----------|
| **Data Parallelism** | Dataset across GPUs; full model replicated | Data doesn't fit batch-wise on 1 GPU |
| **Pipeline Parallelism** | Layers across GPUs | Model layers don't fit on 1 GPU |
| **Tensor Parallelism** | Individual weight tensors across GPUs | Single weights too large for 1 GPU |
| **ZeRO / FSDP** | Optimizer states, gradients, params sharded | Memory-efficient data parallelism |
| **3D Parallelism** | DP + PP + TP combined | Very large models across large GPU clusters |
{:.mbtablestyle}


**References and Image sources:**
* [Distributed communication package - torch.distributed](https://docs.pytorch.org/docs/2.12/distributed.html)
* [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/pdf/1811.06965)
* [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://deepakn94.github.io/assets/papers/pipedream-sosp19.pdf)
* [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053)
* [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054)
* [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
* [PyTorch FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
* [PyTorch FSDP background](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019)