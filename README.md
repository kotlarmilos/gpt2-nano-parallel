# gpt2-nano-parallel

This repository measures five NCCL communication primitives on four RTX 4090
GPUs and fits simple bandwidth and latency models.

The full explanation is in `article.md`. The Hugging Face artifact card is in
`model-card/README.md`.

## Result

The measured release is based on `83d4fbe`.

| Model | Held-out mean absolute relative error | Worst held-out error |
| --- | ---: | ---: |
| One shared model | 22.906% | 71.322% |
| Model per primitive and GPU count | 10.865% | 61.559% |

The measured-composition planner selects the measured winner in 7 of 8
held-out scenarios. Always selecting the context-parallel all-gather proxy
wins 8 of 8. The planner does not beat that fixed baseline.

These are single-node collective microbenchmarks. They are not end-to-end
distributed-training measurements.

## Retained files

- `configs/nccl_publication.toml` is the exact measurement configuration.
- `artifacts/nccl-publication/` contains 80 measured rows and the fitted result.
- `artifacts/manifest.json` records release provenance and SHA-256 hashes.
- `src/gpt2_nano_parallel/gpu_collectives.py` runs and validates the NCCL study.
- `src/gpt2_nano_parallel/topology_planner.py` fits models and ranks declared
  communication compositions.
- `scripts/verify_release.py` verifies hashes, provenance, derived values, and
  the fixed baseline comparison.

## Verify the tracked release

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest -v tests.test_release
python scripts/verify_release.py
```

The verification path runs on CPU. It checks the recorded GPU evidence without
presenting the CPU run as new multi-GPU evidence.

## Reproduce the GPU study

Use one node with four visible CUDA devices.

```bash
python -m pip install -e ".[gpu]"
./scripts/run_nccl_publication.sh
```

A new run should keep its own raw artifacts and fitted result. The checked-in
measurements describe only the recorded host and software environment.

The measured artifacts are also published at
`https://huggingface.co/kotlarmilos/gpt2-nano-parallel`.
