#!/usr/bin/env bash
set -euo pipefail

config="${CONFIG:-configs/nccl_publication.toml}"
run_dir="${RUN_DIR:-../gpt2-nano-parallel-nccl-run}"
output_dir="${OUTPUT_DIR:-artifacts/nccl-publication}"

mkdir -p "$run_dir"

torchrun --standalone --nproc_per_node=2 \
  -m gpt2_nano_parallel.gpu_collectives measure \
  "$config" --output "$run_dir/raw-world-size-2.json"

torchrun --standalone --nproc_per_node=4 \
  -m gpt2_nano_parallel.gpu_collectives measure \
  "$config" --output "$run_dir/raw-world-size-4.json"

python -m gpt2_nano_parallel.gpu_collectives analyze \
  "$config" \
  "$run_dir/raw-world-size-2.json" \
  "$run_dir/raw-world-size-4.json" \
  --output "$run_dir/results.json"

mkdir -p "$output_dir"
cp "$run_dir/raw-world-size-2.json" "$output_dir/raw-world-size-2.json"
cp "$run_dir/raw-world-size-4.json" "$output_dir/raw-world-size-4.json"
cp "$run_dir/results.json" "$output_dir/results.json"
