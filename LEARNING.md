# Deriving five transformer parallelism strategies from first principles

This note builds the five modeled strategies in this repository from ownership,
bytes, and dependency constraints. Every formula below describes an analytical
calculator or a deterministic simulator. None of the ZeRO, tensor, context,
pipeline, or expert results is a measured multi-device benchmark. The repository
keeps that boundary explicit in `README.md` lines 9 to 11, `article.md` lines 3
to 12, and `scripts/generate_artifacts.py` lines 223 to 263.

The publication configuration uses the gpt2-nano shape in
`configs/publication.toml` lines 1 to 15. It sets `44_000_000` parameters, 12
layers, hidden size 512, 8 heads, context length 1024, MLP width 2048, world size
8, 4 pipeline stages, 16 microbatches, 1024 routed tokens, 8 experts, and `top_k`
2. The report builder in `src/gpt2_nano_parallel/cli.py` lines 71 to 113 reads
that configuration and calls the independent calculators for all five strategies.

## Data parallelism with ZeRO

### Motivation

Data parallelism starts from the observation that the loss over a batch is an
average over examples. If each rank owns a different minibatch and a complete
copy of the model, every rank can compute a local gradient. The ranks then sum or
average those gradients before the optimizer step. This increases batch
throughput, but it replicates model state on every rank.

ZeRO changes model-state ownership while leaving the mathematical optimizer step
unchanged. The question becomes which rank must hold each state tensor at a given
moment. If a tensor only needs one owner between collective operations, sharding
that tensor over `N` ranks divides its steady-state memory by `N` in the ideal
balanced model.

### Derivation

Let `P` be the parameter count and `N` be the data-parallel group size. The code
uses the classic mixed-precision Adam accounting by default. Model parameters use
2 bytes, gradients use 2 bytes, Adam moments use 8 bytes total because Adam keeps
two FP32 moment tensors, and master parameters use 4 bytes.

```text
b_param  = 2
b_grad   = 2
b_adam   = 8
b_master = 4
b_total  = b_param + b_grad + b_adam + b_master = 16 bytes per parameter
```

Ordinary data parallelism replicates every component.

```text
M_dp = P * (b_param + b_grad + b_adam + b_master)
```

ZeRO stage 1 shards optimizer state. Parameters and gradients stay replicated,
while the Adam moments and master parameters divide over `N` ranks.

```text
M_zero1 = P * b_param
        + P * b_grad
        + P * b_adam / N
        + P * b_master / N
```

ZeRO stage 2 also shards gradients. Parameters remain replicated because each
rank still executes the full forward and backward pass with local parameters.

```text
M_zero2 = P * b_param
        + P * b_grad / N
        + P * b_adam / N
        + P * b_master / N
```

ZeRO stage 3 shards parameters as well. In the steady-state idealization, every
model-state component scales by `1 / N`.

```text
M_zero3 = P * (b_param + b_grad + b_adam + b_master) / N
```

For the publication shape, `P = 44,000,000` and `N = 8`. These are analytical
model-state bytes, not measured device memory. They exclude activations,
collective workspaces, allocator fragmentation, temporary buffers, and framework
prefetch policy.

```text
M_dp    = 44,000,000 * 16       = 704,000,000 bytes = 671.4 MiB
M_zero1 = 44,000,000 * 2
        + 44,000,000 * 2
        + 44,000,000 * 12 / 8  = 242,000,000 bytes = 230.8 MiB
M_zero2 = 44,000,000 * 2
        + 44,000,000 * 14 / 8  = 165,000,000 bytes = 157.4 MiB
M_zero3 = 44,000,000 * 16 / 8  =  88,000,000 bytes =  83.9 MiB
```

Communication follows from the ring collective model. A ring all-reduce over a
payload of `B` bytes sends `2 * (N - 1) / N` payload bytes per rank. A ring
all-gather or reduce-scatter sends `(N - 1) / N` payload bytes per rank.

```text
V_all_reduce(B, N)     = B * 2 * (N - 1) / N
V_all_gather(B, N)     = B *     (N - 1) / N
V_reduce_scatter(B, N) = B *     (N - 1) / N
```

Data parallelism all-reduces gradients, so its per-rank byte model is

```text
V_dp = P * b_grad * 2 * (N - 1) / N
```

The compact ZeRO communication model in this repository uses one gradient path
and a small number of parameter gather paths. Stage 1 uses the same gradient
all-reduce as data parallelism. Stage 2 uses one gradient reduce-scatter plus one
parameter all-gather. Stage 3 uses one gradient reduce-scatter plus two parameter
gathers.

```text
V_zero1 = P * b_grad * 2 * (N - 1) / N
V_zero2 = P * (b_grad + b_param) * (N - 1) / N
V_zero3 = P * (b_grad + 2 * b_param) * (N - 1) / N
```

For `P = 44,000,000`, `N = 8`, `b_grad = 2`, and `b_param = 2`, the analytical
per-rank volumes are 154,000,000 bytes for data parallelism and ZeRO stage 1,
154,000,000 bytes for ZeRO stage 2, and 231,000,000 bytes for ZeRO stage 3. The
same numeric value for stage 1 and stage 2 is an artifact of this compact byte
model, since both formulas reduce to `P * 4 * 7 / 8`.

### Real Gloo all-reduce model validation

`src/gpt2_nano_parallel/allreduce_validation.py` turns the byte model into a time
model. `analytical_ring_allreduce_seconds` calls
`ring_collective(message_bytes, world_size, "all_reduce")` and divides
`bytes_per_rank` by a fitted effective bandwidth. `ring_latency_steps` adds
`2 * (world_size - 1)` latency steps. `run_gloo_allreduce_sweep` starts local CPU
workers with the Gloo backend and measures real `dist.all_reduce` calls.
`aggregate_rank_measurements` keeps the maximum rank time for each iteration, then
averages those iteration values. `fit_ring_constants` uses least squares on the
configured large messages, and `compare_measurements` records signed and relative
residuals. The artifact uses `configs/allreduce_gloo.toml` and lives at
`artifacts/gloo_allreduce_model_validation.json`.

The run on this host fits `0.513 GiB/s` effective bandwidth and `1,158 microseconds`
per ring latency step from messages at or above `8,388,608` bytes. The largest
messages fit much better than the rest. The `33,554,432` byte rows have `11.3%`
mean relative error, while the full fitted range has `24.1%` and smaller messages
have `121.4%`. Small messages deviate because fixed scheduling, Python process
noise, Gloo algorithm choices, and loopback transport setup dominate the
byte-transfer term. This result is real local communication evidence, but it
validates only CPU Gloo process collectives on one host.

### Where it lives in the code

`src/gpt2_nano_parallel/memory.py` implements the memory derivation. The
`MemoryBreakdown` dataclass stores per-rank `parameters`, `gradients`,
`optimizer`, and `master_parameters` on lines 13 to 23. Its `total` property sums
those fields on lines 24 to 31, and `as_dict` labels the result as `analytical`
on lines 33 to 37. The `zero_memory` function defines the byte defaults on lines
40 to 49, documents BF16 parameters and gradients plus FP32 Adam state on lines
50 to 55, chooses divisors for stages 1, 2, and 3 on lines 70 to 72, and applies
the formulas on lines 73 to 80.

`src/gpt2_nano_parallel/communication.py` implements the communication model.
The `Communication` dataclass labels outputs as `analytical` on lines 13 to 19.
`ring_collective` defines the ring factors for all-reduce, all-gather,
reduce-scatter, and all-to-all on lines 31 to 46. `data_parallel_step` builds the
gradient all-reduce payload on lines 49 to 54. `zero_step` uses the data-parallel
case for stage 1 on lines 72 to 73, computes the shared ring factor on line 74,
computes stage 2 as `parameters * (gradient_bytes + parameter_bytes) * factor` on
line 76, and computes stage 3 as `parameters * (gradient_bytes + 2 *
parameter_bytes) * factor` on line 78.

`scripts/generate_artifacts.py` turns the analytical memory outputs into the SVG
figure. It calls `zero_memory(44_000_000, 8, stage)` on lines 108 to 110 and
writes the label `Analytical, not measured hardware memory` on lines 131 to 140.
The report builder in `src/gpt2_nano_parallel/cli.py` calls `zero_memory`,
`data_parallel_step`, and `zero_step` on lines 81 to 90. The generated
`artifacts/publication_report.json` records the analytical memory values on lines
3 to 43 and the analytical communication values on lines 45 to 69.

## Tensor parallelism

### Motivation

Tensor parallelism starts inside a layer rather than outside the model. A
transformer MLP contains two large linear maps. The first map expands hidden size
`d` to MLP width `h`, and the second map contracts `h` back to `d`. If `h` is
large, a single rank does not need to own every output feature of the first map or
every input feature of the second map.

The strategy splits a matrix dimension across ranks. This reduces parameter,
activation, and compute load for the sharded part of the layer by about `1 / N`.
The cost is synchronization whenever separate partial results must become one
mathematical tensor.

### Derivation

Let `X` be a batch of token activations with shape `T x d`. Let the first MLP
weight be `W1` with shape `d x h`. A column-parallel linear layer partitions the
output columns of `W1`.

```text
W1 = [W1_0, W1_1, ..., W1_{N-1}]
W1_i has shape d x (h / N)
Y_i = X * W1_i
Y_i has shape T x (h / N)
Y = concat_columns(Y_0, Y_1, ..., Y_{N-1})
```

Each rank stores `1 / N` of `W1` and computes `1 / N` of the multiply-adds. If
the next row-parallel layer consumes the sharded `Y_i` directly, the expanded MLP
activation also stays sharded by `1 / N`. The input `X` remains replicated across
the tensor-parallel group in this simple model.

Backward propagation through the column split exposes the first collective. Each
rank can compute its local contribution to the input gradient.

```text
dX_i = dY_i * W1_i^T
dX   = sum_i dX_i
```

The sum requires an all-reduce over a `T x d` tensor. With element size `e`, the
payload is `T * d * e` bytes and the ring per-rank volume is

```text
V_column_backward = T * d * e * 2 * (N - 1) / N
```

The second MLP weight `W2` has shape `h x d`. A row-parallel linear layer
partitions the input rows of `W2` and consumes the sharded activation.

```text
W2 = stack_rows(W2_0, W2_1, ..., W2_{N-1})
W2_i has shape (h / N) x d
Z_i = Y_i * W2_i
Z   = sum_i Z_i
```

Each rank again stores `1 / N` of the weight and computes `1 / N` of the matrix
multiply. The final `Z` has shape `T x d`, and the sum requires an all-reduce over
that full output tensor.

```text
V_row_forward = T * d * e * 2 * (N - 1) / N
```

For the publication shape, `T = 1024`, `d = 512`, `h = 2048`, `N = 8`, and `e =
2`. The rank-local column output has width 256. The rank-local row input also has
width 256. The all-reduce payload is `1024 * 512 * 2 = 1,048,576` bytes, and the
ring model reports `1,048,576 * 14 / 8 = 1,835,008` bytes per rank for both the
column backward and row forward collectives. These are analytical volumes, not
network measurements.

### Where it lives in the code

`src/gpt2_nano_parallel/partitions.py` implements the ownership rules. The
`balanced_ranges` helper divides a length by `world_size` on lines 22 to 34.
`column_partition` validates a rectangular matrix and slices output columns on
lines 47 to 60. `reconstruct_columns` concatenates those column slices on lines
63 to 72. `row_partition` slices input rows on lines 75 to 88, and
`reconstruct_rows` stacks those rows back together on lines 91 to 94.

`src/gpt2_nano_parallel/communication.py` implements the two tensor-parallel
all-reduce payloads in `tensor_linear` on lines 82 to 102. Mode `row` uses
`batch_tokens * out_features * element_bytes` on lines 96 to 98. Mode `column`
uses `batch_tokens * in_features * element_bytes` on lines 99 to 101. Both modes
then call the same analytical ring all-reduce model from `ring_collective` lines
31 to 46.

`src/gpt2_nano_parallel/cli.py` assembles the MLP report in `_tensor_report` on
lines 29 to 68. It records the column projection from `in_features` to
`out_features` on lines 35 to 39, records the row projection from `out_features`
back to `in_features` on lines 40 to 43, emits balanced output and input ranges
on lines 44 to 49, computes column-backward communication on lines 50 to 58, and
computes row-forward communication on lines 59 to 67. The publication report
records the resulting ranges and analytical byte volumes in
`artifacts/publication_report.json` lines 128 to 233.

## Context parallelism

### Motivation

Context parallelism splits the sequence dimension. This targets the memory cost
of sequence-shaped activations. If each rank owns only a contiguous shard of the
context, then tensors shaped like `sequence x hidden` shrink roughly by `1 / N`
per rank.

Attention prevents the split from being fully local. A query at one position can
attend to keys and values at other positions. Therefore the local query shard
needs access to remote key and value shards. The repository models that access
with ring attention byte accounting.

### Derivation

Let the sequence length be `L`, hidden size be `d`, and context-parallel group
size be `N`. A balanced context split assigns each rank about `L / N` token
positions.

```text
L = q * N + r
rank i owns q + 1 tokens if i < r
rank i owns q tokens if i >= r
```

For activations shaped `L x d`, this changes the ideal per-rank activation
storage from `L * d * e` bytes to about `(L / N) * d * e` bytes for each
sequence-shaped tensor. The exact code allows uneven splits by assigning the
remainder to the first ranks.

Attention has a different dependency pattern. For a local query block `Q_i`, full
attention needs every key block `K_j` and every value block `V_j`.

```text
Scores_i = Q_i * K_all^T
A_i      = softmax(Scores_i)
O_i      = A_i * V_all
```

A rank can keep its local queries fixed and circulate key and value blocks around
a ring. The byte count for one block with `t_j` tokens is

```text
B_j = 2 * t_j * d * e
```

The factor of 2 counts `K` and `V`. In a complete ring rotation, each rank sends
`N - 1` blocks. With balanced blocks, the per-rank volume is

```text
V_ring_attention = (N - 1) * 2 * (L / N) * d * e
                 = 2 * L * d * e * (N - 1) / N
```

For uneven blocks, a single average would hide load imbalance. The code reports a
byte count for every rank. It computes the bytes of every block, sums them, and
subtracts the block that the rank does not send under its clockwise convention.

```text
V_rank_i = sum_j B_j - B_{(i + 1) mod N}
```

For the publication shape, `L = 1024`, `d = 512`, `N = 8`, and `e = 2`. Each rank
owns 128 tokens. One local `K,V` block has `2 * 128 * 512 * 2 = 262,144` bytes.
Each rank sends seven such blocks, so the analytical per-rank volume is
`1,835,008` bytes, or 1.75 MiB. This is a communication formula for full
attention. It is not a measured trace, and it does not model causal skipping,
overlap, topology, or kernel fusion.

### Where it lives in the code

`src/gpt2_nano_parallel/partitions.py` implements the sequence split.
`context_partitions` validates positive length and rejects more ranks than token
positions on lines 37 to 44. It delegates the exact range construction to
`balanced_ranges` on lines 22 to 34, where `divmod` creates the base size and
remainder.

`src/gpt2_nano_parallel/communication.py` implements the ring-attention byte
model in `ring_attention` on lines 105 to 140. The docstring states the clockwise
ring convention on lines 112 to 117. The validation is on lines 119 to 123. The
balanced token counts are computed on lines 124 to 127. The `K,V` block bytes use
`2 * tokens * hidden_size * element_bytes` on lines 128 to 130. The per-rank byte
vector subtracts the next the initial block of the next rank from total block bytes on lines
131 to 135, and the return value is labeled `ring_attention_kv` on lines 136 to
140.

`src/gpt2_nano_parallel/cli.py` exposes the context calculator in two places. The
configured report emits `context_partitions` and `ring_attention` on lines 92 to
95. The standalone `context` subcommand labels its output as `analytical` and
prints the same structures on lines 175 to 192. The generated publication report
records the context ranges on `artifacts/publication_report.json` lines 71 to 111
and the analytical ring-attention bytes on lines 113 to 127.

## Pipeline parallelism

### Motivation

Pipeline parallelism splits the model depth. Instead of every rank owning every
transformer block, stage 0 owns the earliest layers, stage 1 owns the next group,
and so on. This reduces per-rank parameter and activation responsibility for the
layer stack, and it lets different microbatches occupy different stages at the
same time.

The cost is idle time. At the beginning, later stages wait for the first
microbatch to arrive. At the end, earlier stages drain before later stages finish.
This idle region is the pipeline bubble.

### Derivation

Let `S` be the number of pipeline stages and `M` be the number of microbatches.
Assume equal stage time and a GPipe forward sweep. At slot 0, only stage 0 can
work. At slot 1, stages 0 and 1 can work. The pipeline becomes full only after
`S - 1` fill slots. It then processes the remaining microbatches and drains.

For one directional sweep, each stage performs `M` useful units of work. The
number of time slots is

```text
slots_direction = M + S - 1
```

The extra `S - 1` slots are the fill and drain overhead measured against the
timeline of one stage. The standard bubble fraction is

```text
bubble = (S - 1) / (M + S - 1)
```

The GPipe schedule runs all forward microbatches and then all backward
microbatches. The backward sweep has the same slot count, so the full simulated
schedule has

```text
slots_total = 2 * (M + S - 1)
```

For the publication shape, `S = 4` and `M = 16`.

```text
bubble = (4 - 1) / (16 + 4 - 1)
       = 3 / 19
       = 0.15789473684210525

slots_total = 2 * 19 = 38
```

More microbatches shrink the fraction because `M` appears in the denominator.
That does not make microbatching free. More microbatches can increase scheduling
work, increase activation bookkeeping, and change optimizer accumulation
behavior. The repository simulates event ordering. It does not measure pipeline
throughput on devices.

### Where it lives in the code

`src/gpt2_nano_parallel/pipeline.py` implements the deterministic GPipe
simulator. The `PipelineEvent` dataclass stores `slot`, `stage`, `microbatch`,
and `phase` on lines 8 to 14. `gpipe_schedule` validates positive dimensions on
lines 16 to 20. It sets `forward_slots = stages + microbatches - 1` on line 22,
then emits forward events with `microbatch = slot - stage` on lines 23 to 28. The
backward sweep starts after the forward slots on line 28 and emits reverse
microbatch order on lines 29 to 35.

The analytical bubble formula lives in `bubble_fraction` on lines 39 to 44. The
report wrapper in `schedule_report` labels the result as `simulated`, records the
scheduler name, copies `stages` and `microbatches`, includes the bubble fraction,
counts total slots, and serializes events on lines 47 to 57.

`src/gpt2_nano_parallel/cli.py` calls `schedule_report` for the configured report
on lines 102 to 105 and for the standalone `pipeline` command on lines 154 to
155. `scripts/generate_artifacts.py` converts the simulated events into the
pipeline SVG on lines 145 to 177 and calls that renderer on lines 187 to 190. The
publication report records `evidence` as `simulated`, the `GPipe
forward-then-backward` scheduler, the bubble fraction, and 38 slots in
`artifacts/publication_report.json` lines 235 to 242.

## Expert parallelism

### Motivation

Expert parallelism splits sparse capacity. A mixture-of-experts layer contains
multiple expert networks, and a gate chooses a small number of experts for each
token. If experts live on different ranks, the model can increase total expert
capacity without making every rank execute every expert for every token.

The price is routing. Tokens start on source ranks, move to destination ranks
that own selected experts, and move back after expert computation. The collective
pattern is all-to-all because many sources can send tokens to many destinations.

### Derivation

Let `R` be the number of token routes. With `T` tokens and `k` selected experts
per token, the route count before capacity filtering is

```text
R = T * k
```

Each route contains one token activation of hidden size `d`. If the element size
is `e`, a remote dispatch route sends

```text
B_route = d * e
```

A local route stays on the same rank and contributes no remote network bytes in
this model. Let `R_remote` be the number of routes whose source rank differs from
the destination rank. Dispatch sends one hidden vector for each remote route, and
combine sends one result vector back.

```text
V_dispatch = R_remote * d * e
V_combine  = R_remote * d * e
V_total    = 2 * R_remote * d * e
```

The repository uses deterministic routing rather than a trained gate. Token
`token` starts on `token mod N`. The first selected expert is a deterministic
permutation of the token index, and top-k routing chooses consecutive experts
from that starting expert. The destination rank is `expert mod N`. This produces
repeatable expert loads and repeatable remote-route counts, which makes the
simulator useful for accounting but not for claims about learned routing quality.

For the publication shape, `T = 1024`, `k = 2`, `d = 512`, `e = 2`, `experts =
8`, and `N = 8`. The simulator creates 2048 routes, counts 1792 remote routes,
and reports

```text
V_dispatch = 1792 * 512 * 2 = 1,835,008 bytes
V_combine  = 1,835,008 bytes
V_total    = 3,670,016 bytes
```

Those are simulated byte counts from a deterministic routing rule. They are not
measured all-to-all timings. Real expert systems also include capacity factors,
padding, token dropping or rerouting, gate noise, load-balancing losses, and
topology-aware dispatch.

### Where it lives in the code

`src/gpt2_nano_parallel/expert.py` implements the expert simulator. The
`TokenRoute` dataclass records token, source rank, expert, and destination rank
on lines 8 to 13. `route_tokens` validates positive inputs, expert divisibility,
and `top_k` on lines 16 to 30. It assigns source ranks with `token % world_size`
on line 33, computes the deterministic first expert on line 34, iterates top-k
choices on lines 35 to 36, and maps experts to destination ranks with `expert %
world_size` on line 37.

`routing_report` computes the accounting on lines 41 to 71. It calls
`route_tokens` on line 52, initializes expert loads and the remote-route counter
on lines 53 to 54, increments expert load and remote routes on lines 55 to 57,
computes `dispatch_bytes = remote_routes * hidden_size * element_bytes` on line
58, labels the result as `simulated` on line 60, and records dispatch, combine,
and total all-to-all bytes on lines 67 to 69.

`src/gpt2_nano_parallel/cli.py` calls `routing_report` in the configured report
on lines 106 to 112 and exposes it as the `experts` subcommand on lines 128 to
133 and 156 to 165. The publication report records the simulated expert
configuration, balanced expert loads, 1792 remote routes, and total all-to-all
bytes in `artifacts/publication_report.json` lines 1013 to 1032. The artifact
manifest marks `pipeline-and-routing` as `simulated` in
`artifacts/manifest.json` lines 20 to 28.

## Combining axes into a device mesh

### Motivation

Large training systems compose these ideas because each one attacks a different
limit. Data parallelism increases the number of examples processed per optimizer
step. ZeRO reduces replicated model state. Tensor parallelism splits wide linear
algebra. Context parallelism splits long sequence activations. Pipeline
parallelism splits depth. Expert parallelism splits sparse capacity.

A device mesh makes that composition explicit. Each physical rank receives a
coordinate, and each coordinate axis defines a process group for one kind of
collective.

### Derivation

Let the mesh sizes be `N_dp`, `N_tp`, `N_cp`, `N_pp`, and `N_ep`. The total number
of devices in the ideal product mesh is

```text
N_total = N_dp * N_tp * N_cp * N_pp * N_ep
```

Each formula above applies to the group size on its own axis, not automatically
to the full product. ZeRO memory divisors use `N_dp`. Tensor-parallel MLP shards
use `N_tp`. Context ranges use `N_cp`. Pipeline bubble uses `N_pp` as `S` and
microbatch count `M`. Expert destination mapping and all-to-all accounting use
`N_ep`.

A combined per-rank state model therefore starts as a product of ownership rules
rather than one universal divisor.

```text
parameter_memory_per_rank roughly follows ZeRO and tensor and pipeline ownership
activation_memory_per_rank roughly follows tensor and context and microbatch scheduling
communication_per_step is the sum of collectives on active mesh axes
```

The word roughly matters. Real systems overlap collectives, checkpoint
activations, prefetch ZeRO-3 parameters, fuse tensor-parallel operations, and
route experts with capacity constraints. The repository intentionally avoids
pretending that this composition has been measured. It gives exact formulas and
small deterministic simulations for each axis in isolation.

### Where it lives in the code

The code models axes independently rather than implementing a real multidimensional
mesh. `configs/publication.toml` lines 9 to 15 provide one `world_size`, one
`pipeline_stages` value, and expert settings. `src/gpt2_nano_parallel/cli.py`
passes that same `world_size` to ZeRO, data-parallel communication, context,
tensor, and expert calculators on lines 75 to 112, while pipeline depth comes
from `parallel["pipeline_stages"]` on lines 102 to 105.

`article.md` states the conceptual composition boundary on lines 166 to 171. It
says data parallelism increases batch throughput, ZeRO reduces replicated model
state, tensor parallelism splits wide layers, context parallelism splits long
sequences, pipeline parallelism splits depth, and expert parallelism splits sparse
capacity. The generated manifest preserves the evidence boundary. Analytical
formula outputs are marked on `artifacts/manifest.json` lines 11 to 19,
deterministic pipeline and routing simulations are marked on lines 20 to 28, and
multi-GPU scaling stays pending on lines 37 to 42.

## Reading the results correctly

The central distinction is evidence type. An analytical result comes from a byte
formula. A simulated result comes from deterministic software events or token
routes. A measured result comes from code that ran on hardware and recorded an
observation.

This repository contains an optional local CPU collective, not a multi-GPU
benchmark. `src/gpt2_nano_parallel/cpu_demo.py` creates local Gloo workers,
performs `dist.all_reduce(value)`, and records measured local timing on lines 20
to 41 and 147 to 159. `scripts/run_cpu_demo.py` writes that optional artifact on
lines 15 to 17. `scripts/generate_artifacts.py` validates that artifact as local
CPU evidence on lines 42 to 105 and explicitly describes it as not a GPU scaling
result on lines 207 to 210. That measured path does not change the status of the
five modeled large-model strategies, which remain analytical or simulated in this
repository.
