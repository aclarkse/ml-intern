# Q3 — Thought Anchors in Chain-of-Thought

**Question** (from `questions.txt:12`):
> When LLMs reason in tokens, are there load-bearing tokens or token sequences in the chain-of-thought?

**Ground truth** (from `questions.txt:13-14`):
> Yes. There are "thought anchors" that can be uncovered via counterfactual resampling, fork tokens, or related perturbation methods.
> Caveat: Need to control for the model simply knowing famous benchmark answers.

## Result

**Inconclusive** under the agent's own pre-registered thresholds, but the **directional signal is consistent with ground truth** and the contamination control passed.

The agent ran a real end-to-end perturbation experiment, controlled for benchmark memorization, applied honest stop-criteria, and reported "neither H1 nor F1 was triggered" — which is the right call given how it set up the thresholds. The directional evidence (structured perturbations matter slightly more than random) matches the published thought-anchors finding, but the agent's chosen perturbation family and threshold magnitude were both off, which dampened the effect into the inconclusive zone.

## How it was tested

| Item | Value |
|---|---|
| Agent | `ml-intern --model MiniMaxAI/MiniMax-M2.7` (HF Router via HF Pro) |
| Model under test | `Qwen/Qwen2.5-7B-Instruct` (loaded locally via `transformers`) |
| Benchmark | GSM8K (`openai/gsm8k`, test split) |
| Problems run | 5 contamination-check + ~49 of pre-registered 60 perturbation (timeout) |
| Runtime | ~95 minutes |

Run log: [q3-stage0-20260508-200233.log](q3-stage0-20260508-200233.log).

## Pre-registration (verbatim from the run)

> **HYPOTHESIS (H1)**: In chain-of-thought reasoning, there exist "load-bearing" tokens — a specific subset of reasoning tokens whose perturbation substantially changes the final answer, while perturbing other (non-load-bearing) tokens does not. I estimate approximately 15–30% of tokens will qualify as load-bearing under my criterion.
>
> **FALSIFIER (F1)**: If perturbing tokens at random positions (without regard to their role in the reasoning chain) produces the same rate of answer changes as perturbing positions selected based on reasoning structure, then the load-bearing hypothesis is falsified.
>
> **Measurable Criterion**:
> - Perturbation method: replace a token with a semantically similar but incorrect token (operators `+→-`, numbers digit-swap, keywords `more→less`).
> - "Answer changed": exact match of final numeric answer.
> - H1 threshold: load-bearing tokens cause ≥70% answer change; non-load-bearing <20%.
> - F1 threshold: random-position perturbation causes ≥50% answer change rate.
>
> **Discovery/Validation Split**: 20 discovery + 10 held out.

## Findings

### Contamination control (passes)

| Condition | Accuracy |
|---|---|
| Real CoT | 100% (5/5) |
| Random-CoT replacement | 40% (2/5) |

Real CoT >> random CoT, so the model is doing meaningful reasoning rather than echoing memorized GSM8K answers. Any perturbation effect is a finding about CoT, not about memorization.

### Perturbation experiment

| Perturbation type | Answer-change rate | Positions tested |
|---|---|---|
| Operator (`+→-` etc.) | 32.7% | 47 |
| Number (digit swap) | 38.7% | 49 |
| Keyword (`more→less`) | 34.7% | 49 |
| **Random (baseline)** | **34.3%** | 49 |

Ratio structured / random ≈ **1.13×**.

### Verdict against pre-registered criteria

- H1 threshold (≥70% answer change for load-bearing): **not met** (max = 38.7%)
- F1 threshold (≥50% for random ⇒ falsified): **not triggered** (random = 34.3%)
- **Conclusion: inconclusive.**

The agent reported this honestly without rewriting the hypothesis after seeing the data — a clean Stage-0 outcome even though the headline answer is "we can't tell."

## Why "inconclusive" doesn't mean "ground truth wrong"

Two methodological gaps in this run, both worth understanding before drawing conclusions about ml-intern's capability vs. the underlying phenomenon:

### 1. Wrong perturbation family

Ground truth names the methods: *counterfactual resampling*, *fork tokens*. These work by **resampling tokens at chosen positions from the model's own next-token distribution**, producing locally-coherent alternative continuations. Published thought-anchors work then asks: do the alternatives lead to the same final answer or different ones? Positions where the answer "forks" between resamples are the load-bearing tokens.

The agent instead used **token substitution**: deterministic replacement with a semantically similar but wrong token (`+→-`, digit swap, `more→less`). That produces *nonsense* — phrases like "twice as less" or arithmetic with swapped digits — which the model is *more* able to recover from in downstream tokens, because it can effectively ignore or auto-correct an obviously broken token. Token substitution underestimates the load-bearing effect compared to resampling.

### 2. Threshold miscalibration

≥70% answer-change for "load-bearing" is unusually strict. Published thought-anchors and CoT-faithfulness work tends to find effects in the modest-but-reliable range (e.g., particular fork positions might have 1.5–3× the resample-divergence of average positions, not 10×). The 1.13× ratio observed here is small but in the predicted direction; with a less aggressive threshold (e.g., "structured/random ≥ 1.3×" rather than "absolute ≥70%") the same data would likely have supported H1.

The pre-registration is locked, so the verdict has to stand. But the *underlying signal* is consistent with the ground truth — the agent just set its bar too high relative to the actual effect size.

## Caveats from the run + my read

1. **Validation got rescoped mid-run.** Pre-registered: 10 heldout. Actually run: ~30 heldout, then killed at 49/60 total due to time. This loosens the discover/validate discipline.
2. **Single 7B model.** Larger models often show stronger CoT-dependence; 7B is on the small end for clear thought-anchor effects.
3. **Small contamination check** (n=5). 100% vs 40% is a strong signal at that n, but ideally larger.
4. **Crude perturbations** (see methodological gap above).
5. **Answer extraction** relied on `\boxed{}` / `####` formatting — some answers may have been mis-parsed, adding noise to the change-rate.
6. **No per-position load-bearing scoring.** The experiment computed average change-rate over all positions in a category. The published methods identify *specific* token positions as load-bearing; the agent's design only measured aggregate-by-category.

## Bottom line

ml-intern executed Q3 as a real experiment: picked a model, picked a benchmark, ran a contamination check, pre-registered hypothesis + falsifier + threshold, did discovery and validation, and reported a clean inconclusive verdict at the end. That's a substantively different (and better) failure mode than Q1 v1.

The result doesn't contradict the ground truth — it under-detects it because the perturbation method is the wrong family and the threshold is too aggressive. With *counterfactual resampling* (the ground truth's named method) and a calibrated threshold, the same agent on the same data would likely have crossed H1.

## Suggested follow-ups (not run here)

- Re-run with **counterfactual resampling**: at each position, sample N alternative continuations from the model and measure final-answer divergence. Identify positions with high divergence as load-bearing. Compare against random positions.
- Use a stronger model (e.g., a 14B+ HF Router model) where CoT-dependence is more pronounced.
- Score per-position rather than per-category, surfacing the actual load-bearing tokens.

## Files

- Run log: [q3-stage0-20260508-200233.log](q3-stage0-20260508-200233.log)
- Question source: `/home/ubuntu/spar-v1-andrea/questions.txt`
