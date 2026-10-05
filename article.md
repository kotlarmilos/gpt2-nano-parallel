---
title: "Predicting GPU communication time from measured collectives"
thumbnail: /blog/assets/gpu-collective-timing/thumbnail.png
authors:
- user: kotlarmilos
---

# Predicting GPU communication time from measured collectives

## TL;DR

I measured five GPU communication primitives on one host with four RTX 4090
GPUs. I then tested whether one shared bandwidth and latency model could predict
all five primitives.

It could not. Models fitted by primitive and GPU count reached 10.865% held-out
mean absolute relative error. One shared model reached 22.906%. The worst error
for the primitive-specific models was still 61.559%, so the fitted model is an
estimator rather than a reliable bound.

I also used the fitted models to rank six declared communication compositions.
The planner selected the measured winner in 7 of 8 held-out scenarios. A fixed
choice of the context-parallel all-gather proxy won all 8 scenarios. The planner
therefore did not beat that simple baseline.

## The question

Parallel training methods use different communication operations. Data
parallelism commonly uses all-reduce. ZeRO can combine reduce-scatter and
all-gather. Tensor parallelism often uses repeated all-reduces. Expert
parallelism uses all-to-all. Pipeline stages exchange tensors with neighboring
stages.

Byte counts alone do not determine communication time. Primitive type, message
size, GPU count, topology, and the communication library all affect the result.
The narrow question in this study is whether one simple timing model is enough
for all five measured primitives on one host.

## The timing model

The model separates a transfer term from a startup term.

$$
\widehat{T}
= \frac{\text{modeled bytes per rank}}{\text{effective bandwidth}}
+ \text{modeled communication steps}\times\text{effective latency}
$$

The byte and step counts depend on the primitive and world size. The model uses
ring byte factors for all-reduce, reduce-scatter, all-gather, and all-to-all.
Neighbor exchange uses one payload and one latency step.

These formulas are modeling choices. They do not claim that NCCL used one
specific internal algorithm. The fitted bandwidth and latency values summarize
the measured behavior under this model. They are not hardware specifications.

I compare two fits.

| Fit | Parameters |
| --- | --- |
| Shared | One bandwidth and latency pair for every primitive and GPU count |
| Primitive-specific | One pair for every primitive and GPU count |

The primitive-specific fit has more parameters. The comparison therefore uses
message sizes that were not used during fitting.

## What was measured

The recorded host contains four NVIDIA GeForce RTX 4090 GPUs. The topology has
`NODE` and `PHB` PCIe paths in one NUMA domain and no NVLink path. The software
environment records PyTorch 2.4.1 with CUDA 12.4 and NCCL 2.20.5.

The five measured primitives are all-reduce, reduce-scatter, all-gather,
all-to-all, and simultaneous neighbor exchange. Neighbor exchange is a small
pipeline communication proxy. It is not a complete pipeline schedule.

| Setting | Value |
| --- | --- |
| GPU counts | 2 and 4 |
| Primitive types | 5 |
| Calibration sizes | 4 KiB, 64 KiB, 1 MiB, 16 MiB |
| Held-out sizes | 16 KiB, 256 KiB, 4 MiB, 64 MiB |
| Warmup calls per setting | 5 |
| Timed calls per setting | 20 |
| Measurement rows | 80 |

Each primitive, GPU count, and message size produces one row. Forty rows fit the
models and forty rows evaluate them. The calibration and held-out sizes do not
overlap. The 64 MiB held-out size is larger than every calibration size, so part
of the evaluation tests extrapolation.

The raw artifacts preserve the per-iteration samples, rank means, checksums,
software versions, GPU inventory, topology, source revision, configuration
hash, and split label.

## Prediction error

For measured time `T` and prediction `T_hat`, absolute relative error is
`abs(T_hat - T) / T`.

| Model | Held-out mean error | Worst held-out error |
| --- | ---: | ---: |
| One shared bandwidth and latency pair | 22.906% | 71.322% |
| Separate pair per primitive and GPU count | 10.865% | 61.559% |

Fitting by primitive and GPU count reduces average held-out error by 12.041
percentage points. The result supports using separate communication models on
this host.

The large worst-case error matters. A lower mean does not make every prediction
accurate. The measured primitive-specific bandwidth values range from 2.65 to
4.52 GiB/s. Fitted latency values range from 4.48 to 72.45 microseconds. These
values should not be copied to another machine without new measurements.

## The measured-composition planner

The planner sums predicted times for declared lists of measured primitives.
It evaluates six labels at the same message size.

| Label | Declared communication composition |
| --- | --- |
| Data parallel | One all-reduce |
| ZeRO-2 | One reduce-scatter and one all-gather |
| Tensor parallel | Two all-reduces |
| Context parallel | One all-gather |
| Pipeline parallel | One neighbor exchange |
| Expert parallel | Two all-to-alls |

These rows are not equivalent training steps. They are explicit communication
compositions. The context-parallel row is an all-gather proxy rather than Ring
Attention. The pipeline row omits pipeline scheduling and computation.

For each held-out GPU count and message size, the planner ranks the compositions
with fitted times. The measured ranking adds the corresponding held-out
primitive times.

The fitted planner selects the measured winner in 7 of 8 scenarios. It misses
the four-GPU 16 KiB scenario. The fitted model selects data parallelism while
the measured composition selects the context-parallel all-gather proxy.

Every measured scenario selects that same all-gather proxy. A fixed rule that
always selects it wins 8 of 8 scenarios. The 7 of 8 result does not establish a
useful strategy-selection advantage. The stronger result is the reduction in
held-out timing error.

## Limits

This study measures isolated NCCL communication on one node. It does not measure
model computation, overlap, optimizer execution, memory pressure, kernel fusion,
convergence, or end-to-end training throughput.

Adding isolated communication times does not reproduce a training step.
Contention and scheduling can change the result when collectives run inside a
model. The study also evaluates only two GPU counts and one physical topology.

The simple model can describe the recorded trend while missing individual
measurements by a large amount. A production planner would need uncertainty,
fresh calibration, workload-specific compositions, and end-to-end validation.

## Code and verification

`src/gpt2_nano_parallel/gpu_collectives.py` measures the five primitives and
reconstructs the analyzed result. `topology_planner.py` contains the byte model,
fits, held-out comparison, composition definitions, and fixed-choice baseline.

The exact configuration is `configs/nccl_publication.toml`. The two raw files
and the analyzed result are in `artifacts/nccl-publication/`.
`artifacts/manifest.json` records their SHA-256 hashes and provenance.

The tracked release can be verified on CPU.

```bash
python -m unittest -v tests.test_release
python scripts/verify_release.py
```

The GPU measurement workflow requires four visible CUDA devices.

```bash
python -m pip install -e ".[gpu]"
./scripts/run_nccl_publication.sh
```

The measured release is based on `83d4fbe`. The raw measurements record source
revision `2b58542`. A new host should produce a new set of raw artifacts rather
than reuse these fitted constants.

## Conclusion

One shared communication model is too coarse for these measurements.
Primitive-specific fits reduce average held-out error from 22.906% to 10.865%.
That improves a small timing estimator, but the remaining worst-case error is
large.

The planner result is weaker. It scores 7 of 8 while a fixed all-gather proxy
choice scores 8 of 8. This experiment supports per-primitive calibration on the
recorded host. It does not show that the planner chooses better distributed
training strategies.

## References

1. ZeRO. *Memory Optimizations Toward Training Trillion Parameter Models*.
   arXiv:1910.02054.
2. Megatron-LM. *Training Multi-Billion Parameter Language Models Using Model
   Parallelism*. arXiv:1909.08053.
3. Ring Attention. arXiv:2310.01889.
4. GPipe. *Efficient Training of Giant Neural Networks using Pipeline
   Parallelism*. arXiv:1811.06965.
5. Switch Transformers. arXiv:2101.03961.

The code and measured evidence are available in
[gpt2-nano-parallel on GitHub](https://github.com/kotlarmilos/gpt2-nano-parallel)
and
[gpt2-nano-parallel on Hugging Face](https://huggingface.co/kotlarmilos/gpt2-nano-parallel).
