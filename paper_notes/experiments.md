# Experiment Log

## Notebook 1: BFCL Subset Construction

Goal:
- create a reproducible BFCL subset for hardware efficiency experiments

Design:
- source dataset: gorilla-llm/Berkeley-Function-Calling-Leaderboard
- target subset size: 100 tasks
- random seed: 188
- saved subset: data/bfcl_subset.json
- saved metadata: data/bfcl_subset_metadata.json

Reason:
- fixed benchmark subset ensures fair comparisons across T4, L4, RTX 6000 Ada, and future TPU experiments
- BFCL is appropriate because it targets LLM function/tool-calling accuracy

## T4 inference run

Hardware:
- Google Colab T4 GPU

Model:
- Qwen/Qwen2.5-7B-Instruct

Backend:
- Hugging Face Transformers

Dataset:
- 100-task stratified BFCL v3 subset
- 25 tasks each from simple, multiple, parallel, and parallel_multiple

Run result:
- 100/100 tasks completed without errors
- mean latency: 52.15 s
- median latency: 39.88 s
- mean tokens/sec: 1.47
- mean peak memory: ~13.23 GB

Category-level latency:
- simple: ~27.98 s
- multiple: ~25.82 s
- parallel: ~76.93 s
- parallel_multiple: ~77.88 s

Initial observation:
- parallel and parallel_multiple tasks are substantially slower than simple/multiple tasks, likely because they require longer outputs and more complex tool-call generation.