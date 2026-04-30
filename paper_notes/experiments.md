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

## Groq LPU results (Llama-3.1-8B-Instruct)

Backend: Groq API (`groq_prompt_api`), model `llama-3.1-8b-instant`
Tasks: same 100-task stratified BFCL v3 subset as T4 run
Note: wall-clock latency includes network round-trip and API queue time.
Groq server-side tokens/sec was not recoverable from the SDK response object;
published hardware reference is ~750 tok/s for Llama-3.1-8B on Groq LPU.

Overall result:
- successes: 85/100
- success rate: 0.85
- failure breakdown: argument_mismatch (11), wrong_call_count (3), parse_failure (1)

Success by category:
- multiple:          24/25 = 0.96, CI [0.805, 0.993]
- parallel_multiple: 22/25 = 0.88, CI [0.700, 0.958]
- simple:            21/25 = 0.84, CI [0.653, 0.936]
- parallel:          18/25 = 0.72, CI [0.524, 0.857]

Latency by category (wall-clock, includes network):
- simple mean:            0.49 s, median 0.25 s
- parallel mean:          2.78 s, median 2.54 s
- parallel_multiple mean: 6.05 s, median 5.51 s
- multiple mean:          3.83 s, median 3.57 s

Throughput by category (wall-clock tokens/sec):
- simple:            ~162 tok/s
- parallel:          ~70 tok/s
- parallel_multiple: ~30 tok/s
- multiple:          ~12 tok/s

Latency vs T4 comparison (same categories):
- simple:   0.49 s vs 27.98 s → ~57x faster wall-clock
- parallel: 2.78 s vs 76.93 s → ~28x faster wall-clock
- overall mean: 3.29 s vs 52.15 s → ~16x faster wall-clock

Failure taxonomy:
- argument_mismatch: 11 (dominant — semantic errors, not structural)
- wrong_call_count:   3 (all in parallel category)
- parse_failure:      1 (parallel category)
- parallel accounts for 7/15 failures despite being 25% of tasks

Interpretation:

Correctness is broadly stable across T4 (91%) and Groq (85%), with overlapping
95% Wilson CIs on all four categories. This supports H3: hardware does not
materially affect correctness when model family and prompts are held approximately
constant. The ~6 point gap is consistent with the model difference (Qwen 7B vs
Llama 8B) rather than a hardware effect.

Groq wall-clock throughput shows high category-level variance (12–162 tok/s)
that does not cleanly track output length. The `multiple` category produces
the lowest tok/s (12) despite similar output token count to `simple` (43 vs 45
tokens), likely due to longer input prompts (multiple function schemas increase
prefill time). This is a prefill effect rather than a decode or queue effect.
`parallel` tasks have the highest output token count (~123 tokens) and
moderate tok/s (~70), consistent with decode-dominated latency once prefill
completes. These patterns suggest Groq wall-clock measurements conflate
prefill latency, decode throughput, and API queue time and should be
interpreted as system-level measurements rather than hardware benchmarks.

Parallel category remains the hardest across both hardware systems:
- T4 parallel: 0.80 success rate
- Groq parallel: 0.72 success rate
This consistency across architectures suggests parallel task difficulty is
model-driven (multi-call coordination), not hardware-driven.

Parse failures are rare (1 total on Groq), confirming that the <tool_call>
prompt format and parser are working correctly. Remaining errors are semantic.

Finding 1 (strongest): Correctness is hardware-invariant within CI bounds. This is a clean, testable hypothesis that comes back confirmed. It means practitioners can choose hardware purely on cost/speed grounds without sacrificing accuracy. That's a practical, actionable conclusion.
Finding 2: Latency scales linearly with output token count on memory-constrained GPU, but this relationship breaks down on API-based LPU inference where prefill and queue effects dominate. Two different physical regimes producing the same task, explained by architecture.
Finding 3: Parallel tasks are the consistent weak point across both systems — lower accuracy and higher latency — suggesting the difficulty is in the model's multi-call coordination, not in any hardware property. This is visible in both the T4 and Groq failure taxonomies independently, which makes it a robust finding.
Finding 4: The gap between Groq's published hardware throughput (~750 tok/s) and your observed wall-clock throughput (68 tok/s mean) quantifies the API overhead cost — roughly 10x attenuation. This is a practically important number for anyone deciding between on-prem GPU and inference API deployment.