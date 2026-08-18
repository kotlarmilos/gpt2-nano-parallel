---
title: "Calibrating GPU collective models for parallelism planning"
thumbnail: /blog/assets/five-parallelism-strategies/thumbnail.png
authors:
- user: kotlarmilos
---

# Calibrating GPU collective models for parallelism planning

## TL;DR

My gpt2-nano model is small enough to make parallel training assumptions
inspectable. This project derives five parallelism axes, then tests a narrower
question on four RTX 4090 GPUs. Can one shared bandwidth and latency model predict
all of the collectives used by those strategies? It cannot. Primitive-specific
models reduce held-out mean relative error from `22.91%` to `10.87%`. A
communication-only planner then selects the measured winner in `7` of `8` held-out
scenarios. This is collective microbenchmark evidence, not end-to-end distributed
training evidence.

## The problem

The model has about 44 million parameters, 12 transformer blocks, a hidden size of
512, 8 attention heads, a context length of 1024, and an MLP width of 2048. I use
those dimensions throughout the examples. The repository contains calculators and
deterministic simulators, so each number can be regenerated. The five methods solve
different limits. Data parallelism increases batch throughput. ZeRO reduces
replicated model state. Tensor parallelism splits wide layers. Context parallelism
splits long sequences. Pipeline parallelism splits depth. Expert parallelism splits
sparse capacity. Large training systems combine them, but each axis is easier to
understand when its ownership and communication rules run in a small, inspectable
example first.

## Intuition

### Data parallelism and ZeRO

Data parallelism starts with the simplest split. Every rank gets a different batch
and a complete model. After backward, the ranks average gradients with an
all-reduce. The model code barely changes, which is why this is usually the first
method to try. The memory cost is replication. ZeRO changes who owns those states.
ZeRO stage 1 shards optimizer moments and master parameters. ZeRO stage 2 also
shards gradients. ZeRO stage 3 shards parameters, gradients, and optimizer state.

### Tensor parallelism

Tensor parallelism divides work inside a layer. A column-parallel linear layer
splits the output columns. Every rank receives the same input activation and
computes its slice of the output, and backward needs a sum across ranks. A
row-parallel linear layer makes the complementary split. The input features and
weight rows are divided across ranks and an all-reduce sums the partials. The two
layouts fit together well. The first projection can produce sharded features
directly for the second projection, which avoids gathering the large intermediate
activation between the two linear layers.

### Context parallelism and ring attention

Context parallelism divides sequence positions. For a context length of 1024 and
eight ranks, every rank owns 128 consecutive tokens. Attention makes this split
less local than it first appears. A query block still needs keys and values from
other token blocks. Ring attention keeps the local queries in place and circulates
key and value blocks around a ring. After eight steps, each query block has visited
every key and value block. The important invariant is coverage. Splitting a
sequence must not lose, duplicate, or reorder positions.

### Pipeline parallelism

Pipeline parallelism assigns groups of transformer blocks to stages. A four-stage
gpt2-nano pipeline can place three of the 12 blocks on each stage. Microbatches then
flow through those stages like independent jobs. The simulator uses the GPipe
schedule. It runs all forward microbatches and then all backward microbatches.
Pipeline stages wait while the pipeline fills and drains.

### Expert parallelism

Mixture-of-experts layers add another ownership axis. Experts are distributed
across ranks, while a gate selects one or more experts for each token. Tokens must
move to the selected expert and return after computation. Real expert systems also
enforce capacity. Load balance matters because the busiest expert determines when
the collective can move on.

## The math

### Data parallelism and ZeRO

With mixed precision AdamW, a useful accounting model assigns 2 bytes to each
parameter, 2 bytes to each gradient, 8 bytes to the two FP32 Adam moments, and 4
bytes to an FP32 master parameter. That is 16 bytes per parameter before
activations and temporary buffers. At 44 million parameters, ordinary data
parallelism holds about 671 MiB of model state on every rank. With eight ranks,
ZeRO stage 1 falls to about 230.8 MiB per rank, stage 2 reaches about 157.4 MiB,
and stage 3 reaches about 83.9 MiB in the ideal balanced model. Those are
analytical values. They do not include activations, allocator fragmentation,
collective workspaces, or parameters that a framework keeps resident for
prefetching. A ring all-reduce sends twice the payload multiplied by the factor of
world size minus one over world size per rank. ZeRO stage 2 can replace the gradient
all-reduce with a reduce-scatter, then gather updated parameters. ZeRO stage 3
gathers parameters as layers need them and reduce-scatters gradients.

### Tensor parallelism

Consider the first gpt2-nano MLP projection from 512 to 2048 features. A
column-parallel split gives eight ranks 256 output features each. For the second MLP
projection, eight ranks partition the 2048 input features and each rank produces a
partial 512-wide output. Tensor parallel groups need compatible dimensions. A width
of 512 divides cleanly across eight ranks. Invalid splits are rejected rather than
silently creating empty partitions.

### Context parallelism

For gpt2-nano, one rank starts with 128 tokens times 512 hidden values for each of K
and V. With 2-byte elements, circulating both tensors through seven peers accounts
for 1.75 MiB transmitted per rank in the simple full-attention model. That number is
analytical. Causal masking can skip work, and real kernels can overlap transfer
with score computation. Uneven contexts also produce uneven communication. The
calculator returns one byte count for every rank instead of labeling an average as
a per-rank value.

### Pipeline parallelism

For equal stage times, the standard bubble fraction is the number of stages minus
one divided by microbatches plus stages minus one. Four stages with 16 microbatches
have a bubble fraction of 3 divided by 19, or about 15.8 percent, for each
directional sweep. More microbatches reduce that fraction, but they can increase
activation storage and scheduling overhead. Uneven block costs also create idle time
that the simple formula cannot see.

### Expert parallelism

For a gpt2-nano hidden vector, one remote route moves 512 values times 2 bytes in
each direction. Top-2 routing creates two expert assignments per token, so the
accounting counts both. A remote route sends one hidden vector during dispatch and
one during combine.

## In the code

The memory accounting for data parallelism and every ZeRO stage lives in
`src/gpt2_nano_parallel/memory.py`. The collective payload formulas live in
`src/gpt2_nano_parallel/communication.py`. Tensor and context partition planners
with their coverage proofs live in `src/gpt2_nano_parallel/partitions.py`. The GPipe
schedule simulator lives in `src/gpt2_nano_parallel/pipeline.py`, and the expert
routing simulator lives in `src/gpt2_nano_parallel/expert.py`. The command line
entry point is `src/gpt2_nano_parallel/cli.py`, and the optional real all-reduce
demo is `src/gpt2_nano_parallel/cpu_demo.py`.
`src/gpt2_nano_parallel/gpu_collectives.py` measures NCCL primitives and keeps
calibration sizes separate from held-out sizes.
`src/gpt2_nano_parallel/topology_planner.py` fits the models and evaluates declared
collective compositions. The tests reconstruct every partition, verify exact
sequence coverage, check each pipeline dependency, and recompute every published
NCCL result from hash-bound raw measurements.

## Results and honesty boundary

There are three useful kinds of output here. Analytical output comes directly from
byte and bubble formulas. Simulated output comes from deterministic schedules and
token routes. Measured output is reserved for code that actually ran. The generated
schedule figure is a deterministic simulation, not a trace collected from workers.
The reported expert loads make imbalance visible without pretending that a
deterministic toy gate matches a trained router.

The optional PyTorch demo launches local CPU processes with Gloo and performs a
real scalar all-reduce. Its elapsed time is a local measurement of that run. It is
not evidence for GPU speed or multi-node scaling. The NCCL study is separate
single-node multi-GPU evidence. This boundary is deliberate. Parallel training is
already difficult to reason about without mixing a clean formula, a simulated
schedule, a collective microbenchmark, and an end-to-end training benchmark into
the same claim.

Extension result. `gpt2-parallel-allreduce` now runs a real local CPU Gloo sweep for
world sizes `2`, `4`, and `8` and writes
`artifacts/gloo_allreduce_model_validation.json`. The sweep averages 12 measured
`dist.all_reduce` calls after 3 warmup calls for message sizes from `4` bytes to
`33,554,432` bytes. Fitting the analytical ring model from `ring_collective` on the
`8,388,608` byte and `33,554,432` byte messages gives an effective bandwidth of
`0.513 GiB/s` and a per-step latency of `1,158 microseconds`. The model fits the
largest `33,554,432` byte messages with `11.3%` mean relative error, but the full
fitted range has `24.1%` mean relative error and small messages have `121.4%` mean
relative error. This is a mixed result rather than a confirmation. It validates the
measurement harness and shows that local CPU Gloo does not follow one clean ring
bandwidth plus latency model across the whole sweep. It still does not measure GPU
or multi-node scaling.

Publication extension. The NCCL study measures `80` rows across world sizes `2`
and `4`, five primitives, and eight message sizes. Four sizes calibrate each model
and four disjoint sizes evaluate it. The host contains four RTX 4090 GPUs in one
NUMA domain. Its recorded topology exposes `NODE` and `PHB` PCIe paths and no
NVLink path.

One shared ring model reaches `22.91%` mean absolute relative error over the `40`
held-out rows. Fitting one effective bandwidth and latency pair per primitive and
world size lowers that error to `10.87%`. The fitted effective bandwidths range
from `2.65` to `4.52 GiB/s`, while fitted latency ranges from `4.48` to
`72.45 microseconds`. Those constants describe this host and this harness. They
are not hardware specifications.

The communication-only planner maps data parallelism, ZeRO-2, tensor parallelism,
context parallelism, pipeline parallelism, and expert parallelism to explicit
collective signatures. Its predicted winner matches the measured composition in
`7` of `8` scenarios. It misses the world-size `4`, `16 KiB` case by choosing data
parallelism when the measured winner is the context-parallel all-gather proxy.
Every other scenario selects the same winner as the composed held-out
measurements.

This result supports a narrow claim. Collective type and world size matter enough
that one shared model loses substantial predictive accuracy on this topology.
The planner does not include model compute, communication overlap, optimizer
execution, memory pressure, kernel fusion, convergence, or a full pipeline or ring
attention implementation. It cannot identify the best end-to-end training
strategy.

Expected publication locations, not yet live:

- [GitHub repository](https://github.com/kotlarmilos/gpt2-nano-parallel)
- [Hugging Face Space](https://huggingface.co/spaces/kotlarmilos/gpt2-nano-parallel)

## References

- [ZeRO](https://arxiv.org/abs/1910.02054)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [GPipe](https://arxiv.org/abs/1811.06965)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Runnable code and validation guide](./README.md)
- [Hugging Face Space calculator source](./space/README.md)
