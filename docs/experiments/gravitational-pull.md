# Gravitational Pull Experiment

Testing whether constraints displace or eliminate the uniform hedging attractor.

## Hypothesis

When forced to break the uniform attractor via explicit constraints (e.g., "allocate 5-80% per option"), models don't distribute probability mass according to genuine beliefs. Instead, they pile mass at constraint boundaries—revealing a **residual attractor** toward uniformity.

The uniform attractor isn't eliminated, just displaced.

## Constraint Levels

We sweep through progressively tighter bounds:

| Level | Bounds | Interior Space |
|-------|--------|----------------|
| Loose | 5-80% | Wide |
| Moderate | 10-70% | Medium |
| Tight | 15-60% | Narrow |
| Very Tight | 20-55% | Minimal |

## Usage

### Single Question Sweep

```bash
python experiments/gravitational_pull.py openai --n-trials 10 --verbose
```

### Multi-Question Sweep

```bash
python experiments/gravitational_pull.py openai --multi-question -o data/gp_results.json
```

### Options

| Flag | Description |
|------|-------------|
| `--n-trials N` | Trials per constraint level |
| `--multi-question` | Use all 40 questions from test set |
| `--verbose` | Print detailed per-trial output |
| `-o FILE` | Output JSON file |

## Key Metric: Boundary Mass

Boundary mass = fraction of probability allocated within 3% of min/max bounds:

```python
boundary_mass = sum(p for p in probs if abs(p - min) < 0.03 or abs(p - max) < 0.03)
```

If the gravitational pull hypothesis is correct, boundary mass should **increase** as constraints tighten.

## Results

### 4-Provider Comparison

| Constraint | OpenAI | Gemini | Llama 3.1 | Qwen 2.5 |
|------------|--------|--------|-----------|----------|
| (5,80) | 17.9% | 28.4% | 9.9% | 18.4% |
| (10,70) | 33.2% | 37.0% | 9.8% | 35.0% |
| (15,60) | 32.2% | 29.6% | 8.6% | 38.1% |
| (20,55) | **42.8%** | 30.6% | 16.2% | **37.8%** |

### Gravitational Pull Strength

| Provider | Trend (5,80 → 20,55) | Strength |
|----------|---------------------|----------|
| OpenAI gpt-4o-mini | **+24.9%** | STRONG |
| Qwen 2.5 7B | **+19.3%** | STRONG |
| Llama 3.1 8B | +6.4% | MODERATE |
| Gemini 2.0-flash | +2.3% | WEAK |

## Interpretation

### Strong Pull (OpenAI, Qwen)

Boundary mass increases dramatically as constraints tighten. These models clearly displace hedging to constraint boundaries when squeezed.

### Moderate Pull (Llama)

Lower baseline boundary mass with modest increase. Less prone to boundary piling overall.

### Weak Pull (Gemini)

Starts with highest baseline but doesn't increase under pressure. Hedges at boundaries by default rather than when forced.

## Visualization

Expected pattern for strong gravitational pull:

```
Constraint   Boundary Mass
(5,80)       ████░░░░░░░░░░░░░░░░  18%
(10,70)      ███████░░░░░░░░░░░░░  33%
(15,60)      ███████░░░░░░░░░░░░░  32%
(20,55)      █████████░░░░░░░░░░░  43%
               ↑
         Mass piles at edges
```

## Implications

1. **Anti-uniform constraints don't reveal true beliefs** - They just shift where hedging occurs
2. **Boundary piling is a tell** - High boundary mass suggests constrained hedging, not genuine uncertainty
3. **Provider differences matter** - Different models have different hedging strategies

## Related

- [Belief Probe Baseline](belief-probe.md) - The foundational experiment
- [Alignment Regularizer](alignment-regularizer.md) - Testing if incentives help
- [Gravitational Pull Notes](../labnotes/gravitational-pull-experiment.md) - Detailed lab notes
