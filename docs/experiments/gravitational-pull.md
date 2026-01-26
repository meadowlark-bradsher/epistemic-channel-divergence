# Gravitational Pull Experiment

Testing whether constraints eliminate or merely displace the uniform hedging attractor.

## Hypothesis

When forced to break the uniform attractor via explicit constraints, models don't distribute probability mass according to genuine beliefs. Instead, they pile mass at constraint boundaries—revealing a **residual attractor** toward uniformity.

## What This Experiment Falsifies

- **Claim**: "Anti-uniform constraints reveal true beliefs"
- **Result**: Partially falsified. Constraints reveal *some* variation but also induce boundary piling.

- **Claim**: "The uniform attractor is fundamental to model architecture"
- **Result**: Falsified. Models *can* express non-uniform distributions when constrained.

## What This Experiment Does NOT Claim

- That boundary piling makes constraints useless (they're still informative)
- That all providers show equal gravitational pull (they don't)
- That tighter constraints are always worse (depends on research question)

## Constraint Levels

| Level | Bounds | Interior Space |
|-------|--------|----------------|
| Loose | 5-80% | Wide |
| Moderate | 10-70% | Medium |
| Tight | 15-60% | Narrow |
| Very Tight | 20-55% | Minimal |

## Usage

```bash
# Single question sweep
python experiments/gravitational_pull.py openai --n-trials 10 --verbose

# Multi-question sweep
python experiments/gravitational_pull.py openai --multi-question -o data/gp_results.json

# All providers
python experiments/gravitational_pull.py --all-providers --multi-question
```

### Options

| Flag | Description |
|------|-------------|
| `--n-trials N` | Trials per constraint level |
| `--multi-question` | Use all 40 questions from test set |
| `--verbose` | Print detailed per-trial output |
| `-o FILE` | Output JSON file |

## Key Metric: Boundary Mass

```python
boundary_mass = sum(p for p in probs
                    if abs(p - min_bound) < 0.03
                    or abs(p - max_bound) < 0.03)
```

If gravitational pull exists, boundary mass should **increase** as constraints tighten.

## Key Results

| Provider | (5,80) | (20,55) | Pull Strength |
|----------|--------|---------|---------------|
| OpenAI | 17.9% | 42.8% | **STRONG** |
| Qwen 2.5 | 18.4% | 37.8% | **STRONG** |
| Llama 3.1 | 9.9% | 16.2% | MODERATE |
| Gemini | 28.4% | 30.6% | WEAK |

## Delivers

- F5: Gravitational pull (displaced attractor)
- Provider-specific hedging strategies
- Evidence that constraints don't reveal "true" beliefs

## Related

- [Findings: Gravitational Pull](../findings/gravitational-pull.md)
- [Findings: Uniform Attractor](../findings/uniform-attractor.md)
