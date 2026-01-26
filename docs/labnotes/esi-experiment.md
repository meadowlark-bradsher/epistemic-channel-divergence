# ESI Experiment: Epistemic Sensitivity under Intervention

**Date:** 2025-01-26
**Providers:** OpenAI (gpt-4o-mini), Gemini (gemini-2.0-flash)
**Questions:** 25 (15 ambiguous, 10 easy)
**Variants per item:** 5

## Overview

We extended the belief probe framework with **ESI (Epistemic Sensitivity under Intervention)** testing to measure how stable action (token) and report (stated) beliefs are under semantic-preserving perturbations.

**Interventions tested:**
- **SOC (Skip-One-Char):** Remove characters from word tails (e.g., "programming" → "programing")
- **ORDER:** Shuffle the A/B/C/D option positions (answer content preserved, labels remapped)

**Metrics:**
- **ESI_action:** Mean Hellinger distance of action distributions under intervention
- **ESI_report:** Mean Hellinger distance of reported distributions under intervention
- **ΔJS:** Change in JS(action||report) mismatch under intervention

## Results Summary

### SOC Intervention (Character-level Perturbation)

| Provider | Probe | ESI_action | ESI_report | ΔJS | Ratio |
|----------|-------|------------|------------|-----|-------|
| OpenAI | A (Act→Report) | 0.185 | 0.037 | +0.022 | 5.0x |
| OpenAI | B (Report→Act) | **0.161** | 0.048 | +0.034 | 3.4x |
| OpenAI | C (CoT→Act) | **0.297** | 0.048 | +0.072 | 6.2x |
| OpenAI | D (Introspect) | 0.194 | 0.063 | +0.046 | 3.1x |
| Gemini | A | 0.221 | 0.080 | -0.017 | 2.8x |
| Gemini | B | 0.314 | 0.079 | +0.080 | 4.0x |
| Gemini | C | **0.414** | 0.068 | +0.050 | 6.1x |
| Gemini | D | 0.258 | 0.069 | +0.010 | 3.7x |

### ORDER Intervention (Option Position Shuffling)

| Provider | Probe | ESI_action | ESI_report | ΔJS | Ratio |
|----------|-------|------------|------------|-----|-------|
| OpenAI | A | 0.555 | 0.086 | -0.020 | 6.5x |
| OpenAI | B | 0.531 | 0.090 | -0.026 | 5.9x |
| OpenAI | C | 0.529 | 0.084 | +0.002 | 6.3x |
| OpenAI | D | 0.551 | 0.098 | -0.027 | 5.6x |
| Gemini | A | 0.517 | 0.098 | -0.055 | 5.3x |
| Gemini | B | 0.426 | 0.102 | +0.008 | 4.2x |
| Gemini | C | 0.543 | 0.102 | +0.022 | 5.3x |
| Gemini | D | 0.530 | 0.119 | -0.087 | 4.5x |

## Key Findings

### F1. Action distributions are 3-6x more sensitive than reported beliefs

Across all conditions, ESI_action >> ESI_report:
- **SOC:** Action sensitivity ~0.2-0.4, Report sensitivity ~0.04-0.08
- **ORDER:** Action sensitivity ~0.5, Report sensitivity ~0.09-0.12

**Interpretation:** The token distribution (behavioral belief) is *brittle* to minor prompt perturbations, while the reporting policy acts as a *stabilizer* that produces relatively consistent outputs. This asymmetry suggests the two channels have fundamentally different properties—action reflects moment-to-moment computation, while reporting invokes a learned "safe policy."

### F2. Option order causes 2-3x more action shift than character perturbation

- **SOC ESI_action:** ~0.2-0.3 (OpenAI), ~0.2-0.4 (Gemini)
- **ORDER ESI_action:** ~0.5 (both providers)

**Interpretation:** Models exhibit strong **position bias**—the A/B/C/D labels carry information beyond the content they label. Shuffling option positions causes dramatic redistribution of probability mass, even when the semantic content is preserved. This is a known phenomenon but the magnitude (~0.5 Hellinger distance) is striking.

### F3. CoT (Probe C) amplifies action sensitivity

CoT shows the highest ESI_action under SOC:
- OpenAI: C=0.297 vs A=0.185, B=0.161, D=0.194
- Gemini: C=0.414 vs A=0.221, B=0.314, D=0.258

**Interpretation:** Chain-of-thought reasoning makes the action distribution *more brittle*, not less. The reasoning step introduces an additional source of variance—small changes to the question text propagate through the reasoning chain and amplify at the action point. This reinforces our earlier finding that CoT is a "variance amplifier."

### F4. Probe B (Report→Act) shows most stable action under SOC

For OpenAI with SOC, Probe B has the lowest ESI_action (0.161):
- The anchoring effect (declaring beliefs before acting) provides some stabilization
- This aligns with our earlier finding that Probe B produces best belief-action alignment

### F5. Report stability is remarkably consistent

ESI_report stays in a narrow range (0.04-0.12) regardless of:
- Intervention type (SOC vs ORDER)
- Probe type (A, B, C, D)
- Provider (OpenAI vs Gemini)

**Interpretation:** The reporting policy is highly regularized. It produces similar outputs even when the underlying action distribution shifts dramatically. This is the "decoupling" phenomenon manifesting as differential sensitivity.

### F6. Intervention effects on mismatch (ΔJS) differ by type

- **SOC:** Mostly positive ΔJS (+0.02 to +0.08) — interventions *increase* action-report mismatch
- **ORDER:** Mixed/negative ΔJS (-0.09 to +0.02) — interventions may *decrease* mismatch on average

**Interpretation:** Character perturbations destabilize alignment (action shifts more than report), while option shuffling has a more complex effect—possibly because both action and report are affected by position bias, but in correlated ways.

### F7. Gemini shows higher uniform attractor persistence

With anti-uniform constraint:
- OpenAI: 0% uniform reports
- Gemini: 8-13% uniform reports

Gemini's reporting policy is less responsive to the anti-uniform constraint, maintaining more of the "safe" uniform reporting behavior.

## Implications

1. **Measurement validity:** Token probabilities (logprobs) reflect a highly sensitive, unstable signal. Any claims about "model confidence" based on single measurements should be treated with caution. Repeated measurement under perturbation is needed for robust inference.

2. **Reporting as policy:** Self-reported beliefs are not a direct readout of internal states—they're outputs of a learned reporting policy that has been trained to be stable (perhaps overly so). The stabilization makes reports less informative about the underlying uncertainty.

3. **Position debiasing:** The strong position bias in action distributions (~0.5 ESI under order shuffling) suggests debiasing interventions (like ensembling over option orders) could improve calibration.

4. **CoT fragility:** Chain-of-thought reasoning should not be assumed to improve reliability—it amplifies variance and makes outputs more sensitive to prompt perturbations. Use with caution on ambiguous inputs.

## Follow-up Analyses

### Analysis A: Order-Averaged (Ensemble) Action Beliefs

**Key question:** Does averaging action distributions over option permutations reduce the action-report divergence?

| Provider | Probe | JS(original) | JS(ensemble) | Improvement |
|----------|-------|--------------|--------------|-------------|
| OpenAI | A | 0.401 | 0.209 | **+48%** |
| OpenAI | B | 0.391 | 0.221 | **+44%** |
| OpenAI | C | 0.325 | 0.174 | **+47%** |
| OpenAI | D | 0.385 | 0.196 | **+49%** |
| Gemini | A | 0.406 | 0.169 | **+58%** |
| Gemini | B | 0.264 | 0.155 | **+41%** |
| Gemini | C | 0.277 | 0.122 | **+56%** |
| Gemini | D | 0.423 | 0.184 | **+56%** |

**Average improvement: ~50%**

**Interpretation:** Position debiasing via ensembling cuts the action-report divergence in half. This is a practical intervention: run the model on multiple option orderings, unpermute and average the action distributions, and you get a substantially better-calibrated estimate of "what the model believes."

### Analysis C: Easy vs Ambiguous Questions

| Category | ESI_action | ESI_report | ΔJS | Uniform% |
|----------|------------|------------|-----|----------|
| Easy | 0.381 | 0.060 | +0.003 | 0% |
| Ambiguous | 0.395 | 0.092 | +0.009 | 7.8% |

**Findings:**
- ESI_action is similar across categories (position bias affects both)
- ESI_report is 50% higher for ambiguous questions (0.092 vs 0.060)
- Uniform attractor appears only on ambiguous questions (and only for Gemini)
- Ambiguous questions show slightly higher ΔJS (interventions hurt alignment more)

**Interpretation:** The reporting policy is more variable on ambiguous questions, but still far more stable than the action channel. The uniform attractor is specifically triggered by ambiguous/normative content.

### Analysis D: Gemini Uniform Attractor Persistence

| Provider | Orig Uniform% | Under Intervention |
|----------|---------------|-------------------|
| OpenAI | 0% | 0% |
| Gemini | 9.3% | 10.3% |

**By probe under ORDER (Gemini):**
- Probe D shows +6.4% uniform shift under ORDER intervention
- Other probes show ~1% shifts

**Interpretation:** Gemini's reporting policy is more "sticky" to the uniform attractor, even with the anti-uniform constraint. ORDER intervention slightly increases this stickiness, especially for the introspective probe. This suggests Gemini's "safe reporting" behavior is more deeply embedded.

### Analysis E: Content-Identity Check (Argmax Preservation)

| Provider | Argmax Flip Rate | Ensemble Restores |
|----------|------------------|-------------------|
| OpenAI | 64.4% | 44% |
| Gemini | 52.4% | 56% |

**Findings:**
- Argmax flips in majority of orderings (52-64%)
- Ensemble only restores original argmax ~50% of time
- By category:
  - Easy questions: 67-72% flip rate, lower improvement
  - Ambiguous questions: 39-63% flip rate, higher improvement

**Interpretation:** Position bias is strong enough to flip the modal answer in most orderings. Ensembling doesn't preserve "correctness" in any obvious sense—it trades argmax stability for calibration. This is a feature, not a bug: the single-shot argmax was biased.

### Analysis F: Bootstrap - How Many Orderings Needed?

| K | Mean JS | Δ from K=1 | Marginal Gain |
|---|---------|------------|---------------|
| 1 | 0.393 | — | — |
| 2 | 0.262 | -0.130 | 0.130 |
| 3 | 0.222 | -0.170 | 0.040 |
| 4 | 0.199 | -0.193 | 0.023 |
| 5 | 0.188 | -0.205 | 0.012 |
| 8 | 0.167 | -0.225 | 0.007/step |
| 10 | 0.161 | -0.232 | 0.003/step |

**Key insight:** Most improvement comes from K=2-4. Beyond K=5, diminishing returns.

**Practical recommendation:** K=4 orderings captures ~85% of the benefit at 40% of the cost of K=10.

### Bonus: Action Entropy vs Sensitivity

| Entropy | N | Mean ESI_action |
|---------|---|-----------------|
| Low (<1 bit) | 252 | 0.393 |
| Mid (1-1.5 bits) | 85 | 0.409 |
| High (>1.5 bits) | 57 | 0.343 |

High-entropy (uncertain) actions are slightly *more stable* under intervention than low-entropy (confident) actions. This counters the intuition that "confident answers are robust"—in fact, peaked distributions may be more sensitive to small perturbations.

## Experimental Details

**Command:**
```bash
python experiments/esi_runner.py --providers openai gemini --full --n-variants 5 --max-items 25 \
    -o results/esi/full_experiment.jsonl
```

**Output files:**
- `results/esi/full_experiment.jsonl` — Per-item results
- `results/esi/full_experiment.summary.csv` — Summary statistics

**Code:**
- `experiments/esi_runner.py` — Main experiment runner
- `interventions/soc.py` — Skip-One-Char intervention
- `interventions/option_order.py` — Option order shuffling
