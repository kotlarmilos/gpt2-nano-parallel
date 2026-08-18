# Calibrating GPU collective models for parallelism planning

This release candidate accompanies `article.md`. It turns the communication and
memory formulas into typed Python calculators, deterministic simulations, one
optional local CPU collective, one local Gloo model validation sweep, and one
single-node NCCL topology study. The examples use the public gpt2-nano shape:
about 44 million parameters, 12 layers, hidden size 512, 8 attention heads,
context length 1024, and MLP width 2048.

The tracked release artifacts contain a four-GPU collective microbenchmark. They
do not contain end-to-end distributed training. JSON fields and
`artifacts/manifest.json` label evidence as `analytical`, `simulated`, `measured`,
or `pending`.

## Included implementations

| Area | Implementation | Evidence |
|---|---|---|
| Data parallel and ZeRO 1–3 | Model-state memory and ring communication calculators | Analytical |
| Tensor parallelism | Row and column matrix partition and reconstruction planners | Executable |
| Context parallelism | Exact token-range planner and ring K/V accounting | Analytical |
| Pipeline parallelism | GPipe forward-then-backward event simulator and bubble formula | Simulated |
| Expert parallelism | Deterministic top-k token routing and remote all-to-all accounting | Simulated |
| CPU processes | Optional PyTorch Gloo all-reduce and model validation sweep | Measured locally when run |
| GPU collectives | NCCL calibration, held-out prediction, and communication-only planning | Measured on four RTX 4090 GPUs |

The ZeRO defaults count BF16 parameters and gradients at 2 bytes each, two FP32
Adam moments at 8 bytes total, and FP32 master parameters at 4 bytes. Activation
memory, temporary buffers, fragmentation, and framework overhead are excluded.

## Run

Python 3.11 or newer is required. The core package has no runtime dependencies.

```bash
cd gpt2-nano-parallel
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/gpt2-parallel report configs/smoke.toml
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/generate_artifacts.py
```

For a clean Colab regeneration of the analytical and simulated artifacts:

```bash
cd gpt2-nano-parallel
./scripts/colab_publication.sh
```

This command does not produce multi-GPU measurements. A standard Colab runtime
usually exposes one accelerator, so ZeRO, tensor, context, pipeline, and expert
scaling remain analytical or simulated until they run on a separate multi-GPU
host.

### Measured NCCL topology study

`gpt2_nano_parallel.gpu_collectives` measures all-reduce, reduce-scatter,
all-gather, all-to-all, and simultaneous neighbor exchange with NCCL. The
publication configuration separates calibration message sizes from held-out
validation sizes. It compares one shared ring model with primitive-specific
models and evaluates a communication-only planner against composed held-out
measurements. The runner keeps its intermediate outputs outside the repository
until both world sizes and the analysis finish, then copies the two raw artifacts
and analyzed result into `artifacts/nccl-publication/`.

The publication run measures `80` rows on four RTX 4090 GPUs. Primitive-specific
models reach `10.87%` held-out mean absolute relative error, compared with
`22.91%` for one shared ring model. The communication-only planner matches the
measured composition winner in `7` of `8` held-out scenarios. The one miss occurs
at world size `4` and `16 KiB`.

Run it on one node with four visible CUDA devices:

```bash
python -m pip install -e ".[cpu-demo]"
./scripts/run_nccl_publication.sh
```

The planner maps data parallelism, ZeRO-2, tensor parallelism, context
parallelism, pipeline parallelism, and expert parallelism to declared collective
signatures. It ranks communication only. It does not model compute, overlap,
optimizer execution, kernel fusion, or convergence, so it cannot identify the
best end-to-end training strategy.

### Examples

```bash
.venv/bin/gpt2-parallel memory --parameters 44000000 --world-size 8 --stage 3
.venv/bin/gpt2-parallel tensor --batch-tokens 1024 --world-size 8
.venv/bin/gpt2-parallel context --sequence-length 1024 --world-size 8
.venv/bin/gpt2-parallel pipeline --stages 4 --microbatches 16
.venv/bin/gpt2-parallel experts --tokens 1024 --experts 8 --world-size 8 --top-k 2
```

## Optional measured CPU demo

PyTorch is deliberately optional.

```bash
.venv/bin/python -m pip install -e '.[cpu-demo]'
.venv/bin/python scripts/run_cpu_demo.py
.venv/bin/gpt2-parallel-allreduce configs/allreduce_gloo.toml --output artifacts/gloo_allreduce_model_validation.json
.venv/bin/python scripts/generate_artifacts.py
```

This launches local CPU processes with the Gloo backend and measures a scalar
all-reduce. Process startup and collective timing are real local observations,
but they say nothing about GPU throughput, GPU interconnects, or scaling.

The all-reduce validation command measures real local CPU Gloo collectives across
world sizes `2`, `4`, and `8`, then compares them with the analytical ring byte
model from `src/gpt2_nano_parallel/communication.py`. The local run fitted
`0.513 GiB/s` effective bandwidth and `1,158 microseconds` per ring latency step.
The largest `33,554,432` byte messages had `11.3%` mean relative error. The full
fitted range had `24.1%`, and smaller messages had `121.4%`. This is measured
single-host CPU communication, not multi-GPU scaling.

The release-candidate run completed with PyTorch 2.13.0 and two local CPU
processes. Both ranks produced the expected sum of `3.0`. The measured wall and
per-rank collective times are recorded in
`artifacts/cpu_allreduce_measured.json`; they describe only that local run.

## Artifacts

`scripts/run_cpu_demo.py` writes the measured CPU JSON. The artifact generator
validates its provenance and contents before marking it as measured.

`scripts/generate_artifacts.py` writes:

- `artifacts/publication_report.json`
- `artifacts/figures/zero_memory.svg`
- `artifacts/figures/pipeline_schedule.svg`
- `artifacts/manifest.json`

`gpt2-parallel-allreduce` writes
`artifacts/gloo_allreduce_model_validation.json`. The NCCL publication runner
writes:

- `artifacts/nccl-publication/raw-world-size-2.json`
- `artifacts/nccl-publication/raw-world-size-4.json`
- `artifacts/nccl-publication/results.json`

The SVGs use formula and simulator outputs only. They contain no fabricated
throughput, latency, utilization, or memory measurements.

`space/` is a standalone Hugging Face Gradio Space for the ZeRO calculator. It
is an optional publishing artifact and is not deployed by this repository.

## Limits

The communication formulas model balanced ring collectives and omit contention,
overlap, topology, bucketing, recomputation, and implementation details. The Gloo
validation calibrates one effective local CPU bandwidth and latency pair, but it
does not turn the formula into a GPU or multi-node benchmark. ZeRO-3 communication
uses two parameter gathers plus one gradient reduce-scatter per step as a compact
educational model. Real frameworks can prefetch, retain, or gather parameters at
different granularities.

The measured NCCL study records `nvidia-smi topo -m` and fits each primitive at
world sizes two and four on that one topology. Its neighbor exchange is a
pipeline communication proxy, and its all-gather is a context-parallel proxy.
Neither is a complete training implementation.

Ring-attention output reports an explicit byte count for every rank. Uneven
context partitions therefore remain visible rather than being presented as an
average rank.
