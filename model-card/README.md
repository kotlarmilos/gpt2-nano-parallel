---
pretty_name: "RTX 4090 NCCL collective timing study"
tags:
- nccl
- distributed-training
- benchmarking
---

# RTX 4090 NCCL collective timing study

This artifact contains 80 NCCL measurements from one host with four RTX 4090
GPUs. It covers all-reduce, reduce-scatter, all-gather, all-to-all, and
simultaneous neighbor exchange at world sizes two and four.

Primitive-specific timing models reach 10.865% held-out mean absolute relative
error. One shared model reaches 22.906%. The measured-composition planner
selects the measured winner in 7 of 8 scenarios, while a fixed all-gather proxy
choice wins 8 of 8.

## Files

- `configs/nccl_publication.toml`
- `artifacts/nccl-publication/raw-world-size-2.json`
- `artifacts/nccl-publication/raw-world-size-4.json`
- `artifacts/nccl-publication/results.json`
- `artifacts/manifest.json`

## Intended use

Use these files to inspect the measured study, reproduce its derived values, or
compare another host with the same harness.

## Limits

The measurements cover one node, one topology, and isolated communication.
They do not measure end-to-end training or establish a portable hardware model.
The planner does not beat the fixed all-gather proxy baseline.

## Verification

```bash
python scripts/verify_release.py
```

The implementation is available at
`https://github.com/kotlarmilos/gpt2-nano-parallel`.
