# Uniform Attractor

Models default to uniform probability reports (25/25/25/25) regardless of their actual beliefs.

## The Phenomenon

When asked to report uncertainty without constraints, models overwhelmingly produce uniform distributions:

- **100% uniform reports** across probes (A/B/C/D) without constraint
- Holds across providers (OpenAI, Gemini, Llama, Qwen)
- Holds across prompt orderings and question types
- JSON compliance does not break the attractor

## Evidence

### Unconstrained Reporting

| Provider | Uniform Rate | Example Report |
|----------|--------------|----------------|
| OpenAI gpt-4o-mini | 100% | {"A": 25, "B": 25, "C": 25, "D": 25} |
| Gemini 2.0-flash | 100% | {"A": 25, "B": 25, "C": 25, "D": 25} |
| Llama 3.1 8B | 100% | {"A": 25, "B": 25, "C": 25, "D": 25} |
| Qwen 2.5 7B | 100% | {"A": 25, "B": 25, "C": 25, "D": 25} |

### Anti-Uniform Constraint

A mild constraint (5-80% per option) breaks the attractor:

| Constraint | Uniform Rate | Report Entropy |
|------------|--------------|----------------|
| None | 100% | 2.0 bits (max) |
| 5-80% | 0-2% | ~1.9 bits |

## Interpretation

This is **not ignorance or incapacity**. It is a **policy equilibrium** induced by training and evaluation pressure.

The uniform attractor represents a "safe" output—hedged, non-committal, unlikely to be penalized. RLHF and human evaluation may reward cautious probability expressions, creating a stable basin of attraction around uniformity.

## The Displaced Attractor

When constrained away from uniform, models don't express genuine beliefs. Instead, they pile probability at constraint boundaries:

| Constraint | OpenAI Boundary Mass | Interpretation |
|------------|---------------------|----------------|
| (5, 80) | 18% | Low boundary piling |
| (10, 70) | 33% | Increasing |
| (15, 60) | 32% | Stable |
| (20, 55) | 43% | Heavy boundary piling |

The uniform attractor isn't eliminated—just **displaced to the nearest permissible values**. See [Gravitational Pull](gravitational-pull.md) for full analysis.

## Implications

1. **Default reports are uninformative** - Without constraints, self-reports reveal nothing about beliefs
2. **The attractor is optional** - Models *can* express graded uncertainty when constrained
3. **Constraints reveal hedging strategy** - Boundary piling is a "tell" for displaced uniformity
4. **Training creates the attractor** - This is learned behavior, not fundamental architecture

## Related

- [Gravitational Pull](gravitational-pull.md) - How constraints displace the attractor
- [Belief-Action Decoupling](belief-action-decoupling.md) - The gap between action and report
