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