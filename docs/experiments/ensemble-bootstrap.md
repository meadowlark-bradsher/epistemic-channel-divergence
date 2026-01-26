# Ensemble Bootstrap Experiment

Order-ensemble as a first-class measurement protocol—turning ESI diagnostics into a practical fix.

## Hypothesis

If position bias is a nuisance symmetry, averaging over option orderings should:

1. Reduce action-report divergence (by marginalizing position)
2. Follow diminishing returns (most benefit from small K)
3. Change the modal answer (since single-shot argmax was biased)

## What This Experiment Falsifies

- **Claim**: "Single-shot measurement is sufficient"
- **Result**: Falsified. Ensemble reduces JS divergence by ~50%.

- **Claim**: "More orderings always help"
- **Result**: Partially falsified. K=4 captures ~85% of benefit.

## What This Experiment Does NOT Claim

- That ensembling fixes report decoupling (it doesn't)
- That ensemble argmax is "correct" (it trades bias for calibration)
- That K=4 is optimal for all use cases (cost/benefit tradeoff)

## Method

1. Generate K permutations of option order
2. Run model on each permutation
3. Unpermute results back to canonical labels
4. Average the action distributions

```python
ensemble_action = average([
    unpermute(action_k, mapping_k)
    for action_k, mapping_k in permutations
])
```

## Usage

```bash
# Basic run with K=10 orderings
python experiments/esi_ensemble.py --providers openai gemini

# With bootstrap analysis
python experiments/esi_ensemble.py --providers openai gemini --bootstrap

# Custom K
python experiments/esi_ensemble.py openai -K 4

# Output to file
python experiments/esi_ensemble.py --providers openai gemini --bootstrap -o results/esi/ensemble.jsonl
```

### Options

| Flag | Description |
|------|-------------|
| `--providers` | One or more providers |
| `-K, --n-orderings` | Number of orderings (default: 10) |
| `--bootstrap` | Run bootstrap analysis (JS vs K) |
| `--max-items` | Limit number of questions |
| `-o FILE` | Output file (JSONL) |

## Key Results

### JS Reduction

| Provider | JS(single) | JS(ensemble) | Improvement |
|----------|------------|--------------|-------------|
| OpenAI | 0.397 | 0.175 | **+37%** |
| Gemini | 0.388 | 0.146 | **+51%** |

### Bootstrap: JS vs K

| K | Mean JS | Δ from K=1 | % of Max Benefit |
|---|---------|------------|------------------|
| 1 | 0.393 | — | 0% |
| 2 | 0.262 | -0.130 | 56% |
| 3 | 0.222 | -0.170 | 73% |
| 4 | 0.199 | -0.193 | **83%** |
| 5 | 0.188 | -0.205 | 88% |
| 10 | 0.161 | -0.232 | 100% |

**Recommendation**: K=4 captures ~85% of benefit at 40% of cost.

### Argmax Preservation

| Provider | Argmax Flip Rate | Ensemble Restores Original |
|----------|------------------|---------------------------|
| OpenAI | 64.4% | 44% |
| Gemini | 52.4% | 56% |

Argmax flips in majority of orderings. Ensembling doesn't "preserve" the original—it gives a different (less biased) answer.

## Output Format

```json
{
  "item_id": "openai_0",
  "js_single": 0.42,
  "js_ensemble": 0.19,
  "improvement_pct": 55.2,
  "argmax_single": "A",
  "argmax_ensemble": "B",
  "argmax_flip_count": 7,
  "argmax_flip_rate": 0.7,
  "bootstrap_js_by_k": {
    "2": 0.31,
    "4": 0.22,
    "10": 0.19
  }
}
```

## Delivers

- ~50% JS reduction via position debiasing
- Optimal K≈4 recommendation
- Content-identity analysis (argmax preservation)
- Practical measurement protocol

## Practical Protocol

### For Research

```bash
# Run with K=4, compare single vs ensemble
python experiments/esi_ensemble.py --providers openai -K 4 --bootstrap
```

Report both `js_single` and `js_ensemble` metrics.

### For Production

```python
def get_calibrated_belief(question, provider, K=4):
    permutations = generate_permutations(question, K)
    actions = [provider.get_action(perm) for perm in permutations]
    return average(unpermute(actions))
```

Cost: K× API calls per question.

## Related

- [Findings: Measurement Implications](../findings/measurement-implications.md)
- [ESI Experiment](esi.md) - The diagnostic that motivated this fix
