# LLM Hardware Cost-Performance Study

This project evaluates cost-performance tradeoffs of tool-using LLM inference across different accelerator architectures.

## Goals
- Compare GPUs (T4, L4, RTX 6000 Ada)
- Measure latency, throughput, and correctness
- Compute cost-per-successful-task (CPST)

## Structure
- notebooks/: experiments
- scripts/: reusable code
- data/: benchmark subsets
- results/: outputs

## Hardware
- Colab T4
- NVIDIA L4
- RTX 6000 Ada
- (future) TPU v5e

## Benchmark
- BFCL subset (function-calling tasks)