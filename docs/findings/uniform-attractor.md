# Uniform Attractor

Historical runs in this project suggest that unconstrained probability reports collapse toward
25/25/25/25 regardless of action confidence. The constrained half of this story is backed by
checked-in artifacts today; the unconstrained raw artifact is not yet committed.

## The Phenomenon

Historical local runs indicated near-complete collapse to uniform reports when the reporting
prompt was left unconstrained. For publication and external audit, the important current repo
status is:

- The constrained baseline artifact is committed as `data/experiment2_results.json`
- The unconstrained baseline artifact is now reproducibly wired up as `make unconstrained-baseline`
- Future baseline outputs record parse-health and raw reports so parser failures can be audited

The exact unconstrained uniform rate for the paper should be quoted from the committed
`data/unconstrained_baseline_results.json` artifact once generated.

## Evidence

### Current Checked-In Evidence

| Artifact | Status | What it supports |
|----------|--------|------------------|
| `data/experiment2_results.json` | Committed | Anti-uniform follow-up. Uniform rates in the saved Ollama baseline are 0%, 5%, 2.5%, and 0% across probes A/B/C/D. |
| `data/provider_comparison.json` | Committed | Cross-provider comparison under `anti_uniform=true`, with per-probe `report_parse_ok` fields. |
| `data/unconstrained_baseline_results.json` | Not yet committed | Will be the paper-facing unconstrained baseline with parse-health and raw-report audit fields. |

### Anti-Uniform Constraint

A mild constraint (5-80% per option) breaks the attractor in the checked-in Ollama baseline:

| Constraint | Uniform Rate | Report Entropy |
|------------|--------------|----------------|
| None | Pending committed artifact | Pending recommit |
| 5-80% | 0-5% across probes in `data/experiment2_results.json` | ~1.9 bits |

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

1. **Default reports appear uninformative** - Without constraints, self-reports may collapse to a safe uniform policy
2. **The attractor is optional** - Models *can* express graded uncertainty when constrained
3. **Constraints reveal hedging strategy** - Boundary piling is a "tell" for displaced uniformity
4. **Training creates the attractor** - This is learned behavior, not fundamental architecture

## Related

- [Gravitational Pull](gravitational-pull.md) - How constraints displace the attractor
- [Belief-Action Decoupling](belief-action-decoupling.md) - The gap between action and report
