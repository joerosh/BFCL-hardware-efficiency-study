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

## T4 scoring results

Scored the T4 run on the 100-task BFCL v3 stratified subset.

Overall:
- success rate: 74/100 = 0.74

Success by category:
- simple: 0.72
- multiple: 0.76
- parallel: 0.72
- parallel_multiple: 0.76

Latency by category:
- simple mean latency: ~27.98 s
- multiple mean latency: ~25.82 s
- parallel mean latency: ~76.93 s
- parallel_multiple mean latency: ~77.88 s

Observations:
- The model generally emits parseable Qwen-style `<tool_call>` JSON.
- Most failures are meaningful tool-calling errors, especially missing required arguments or argument mismatches.
- Parallel categories have much higher latency because they generate longer outputs and require multiple function calls.
- Correctness is similar across categories, but latency increases substantially for parallel workloads.

Scoring limitation:
- Current scorer is a simplified exact-match tool-call scorer.
- It compares function names and argument values, allowing list-valued gold alternatives.
- It does not fully reproduce the official BFCL AST evaluator.

## T4 corrected scoring results

After updating the scorer to avoid penalizing omitted optional arguments, the T4 BFCL success rate increased from 74% to 91%.

Corrected overall result:
- success: 91/100
- success rate: 0.910
- 95% Wilson CI: [0.838, 0.952]

Success by category:
- simple: 24/25 = 0.96, CI [0.805, 0.993]
- multiple: 25/25 = 1.00, CI [0.867, 1.000]
- parallel: 20/25 = 0.80, CI [0.609, 0.911]
- parallel_multiple: 22/25 = 0.88, CI [0.700, 0.958]

Latency by category:
- simple mean latency: ~27.98 s
- multiple mean latency: ~25.82 s
- parallel mean latency: ~76.93 s
- parallel_multiple mean latency: ~77.88 s

Failure taxonomy after optional-argument correction:
- argument_mismatch: 8
- wrong_call_count: 1

Failure distribution:
- parallel: 4 argument mismatches, 1 wrong call count
- parallel_multiple: 3 argument mismatches
- simple: 1 argument mismatch
- multiple: 0 failures

Interpretation:
- Most previous failures were due to optional-argument strictness in the simplified scorer.
- After correcting for optional schema fields, remaining errors are mostly true argument-value mismatches.
- Parallel tasks are both slower and less accurate, suggesting that multi-call generation increases difficulty and output length.
- Despite high correctness, T4 + Transformers has very low throughput (~1.47 output tokens/sec), likely due to memory pressure and vanilla generation overhead.
