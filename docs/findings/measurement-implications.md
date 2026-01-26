# Measurement Implications

How to measure model beliefs reliably given the phenomena we've discovered.

## The Problem

Single-shot measurement of LLM beliefs is unreliable because:

1. **Position bias**: Option order affects action distributions (~0.5 Hellinger)
2. **Report decoupling**: Self-reports don't reflect action beliefs (~0.4 JS)
3. **Perturbation sensitivity**: Small prompt changes shift actions (3-6x more than reports)

## The Solution: Order-Ensemble Measurement

Average action distributions over multiple option orderings to marginalize position bias.

### How It Works

1. Generate K permutations of option order
2. Run the model on each permutation
3. Unpermute each result back to canonical labels
4. Average the distributions

```python
def ensemble_action(question, K=4):
    permutations = generate_permutations(question, K)
    distributions = []
    for perm_question, mapping in permutations:
        action = get_action_probs(perm_question)
        canonical = unpermute(action, mapping)
        distributions.append(canonical)
    return average(distributions)
```

## Results: Ensemble Reduces Divergence by ~50%

| Provider | JS(single) | JS(ensemble) | Improvement |
|----------|------------|--------------|-------------|
| OpenAI | 0.401 | 0.209 | **+48%** |
| Gemini | 0.406 | 0.169 | **+58%** |

Average improvement: **~50%**

## Bootstrap: How Many Orderings?

| K | Mean JS | Δ from K=1 | Marginal Gain |
|---|---------|------------|---------------|
| 1 | 0.393 | — | — |
| 2 | 0.262 | -0.130 | 0.130 |
| 3 | 0.222 | -0.170 | 0.040 |
| 4 | 0.199 | -0.193 | 0.023 |
| 5 | 0.188 | -0.205 | 0.012 |
| 8 | 0.167 | -0.225 | 0.007/step |
| 10 | 0.161 | -0.232 | 0.003/step |

**Key insight**: Most improvement comes from K=2-4. Beyond K=5, diminishing returns.

**Practical recommendation**: K=4 orderings captures ~85% of the benefit at 40% of the cost.

## Argmax Preservation

Ensembling changes the modal answer:

| Provider | Argmax Flip Rate | Ensemble Restores Original |
|----------|------------------|---------------------------|
| OpenAI | 64.4% | 44% |
| Gemini | 52.4% | 56% |

Position bias flips the argmax in majority of orderings. Ensembling doesn't "preserve correctness"—it trades argmax stability for calibration.

This is a **feature, not a bug**: the single-shot argmax was biased by position.

## Measurement Protocol

For reliable belief measurement:

### Minimum (Quick Check)

1. Use anti-uniform constraint (5-80%)
2. Use Report→Act probe ordering
3. Trust action more than report

### Recommended (Research Quality)

1. Use K=4 order-ensemble
2. Use anti-uniform constraint
3. Compare action_ensemble to report
4. Report both single-shot and ensemble metrics

### Avoid

- Single-shot action without perturbation check
- Trusting self-reported probabilities
- CoT→Act on ambiguous questions
- Sampling-based action estimation (use true logprobs)

## What Ensembling Does NOT Fix

- Report-action decoupling (reports still dominated by attractors)
- Provider-specific hedging strategies
- CoT variance amplification
- Fundamental lack of introspective access

Ensembling fixes **position bias**. The other phenomena require different interventions or acceptance.

## Implications for Downstream Use

### When Using LLM Confidence

- Don't use self-reported probabilities
- Use action logprobs with order-ensemble
- Report confidence intervals, not point estimates

### When Evaluating Calibration

- Ensemble over orderings before computing calibration metrics
- Compare to random baseline (uniform over orderings)
- Report both single-shot and ensemble calibration

### When Comparing Models

- Same ensemble protocol for all models
- Report position bias magnitude (ESI_action under ORDER)
- Account for different hedging strategies

## Related

- [Structural Sensitivity](structural-sensitivity.md) - Why position bias exists
- [Belief-Action Decoupling](belief-action-decoupling.md) - Why reports can't be trusted
