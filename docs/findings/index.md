# Findings

Authoritative findings and working claims from the epistemic channel divergence research.
Most are backed by committed artifacts; the unconstrained F1 baseline is currently pending
recommit as a paper-grade raw dataset.

## Phenomena

| Finding | Page | Summary |
|---------|------|---------|
| Uniform Attractor | [uniform-attractor.md](uniform-attractor.md) | Historical local runs show collapse toward 25/25/25/25; repo-grade raw artifact pending recommit |
| Belief-Action Decoupling | [belief-action-decoupling.md](belief-action-decoupling.md) | Models act confidently but report uncertainty |
| Probe Order Effects | [probe-order-effects.md](probe-order-effects.md) | Report→Act beats CoT→Act; framing is causal |
| Gravitational Pull | [gravitational-pull.md](gravitational-pull.md) | Constraints displace hedging to boundaries |
| Structural Sensitivity | [structural-sensitivity.md](structural-sensitivity.md) | Actions are 3-6x more sensitive than reports |
| Measurement Implications | [measurement-implications.md](measurement-implications.md) | Ensembling restores invariance |

## The Unified Picture

All observations reduce to a **three-layer system**:

1. **Action system**: Token-level, sensitive, reveals true confidence
2. **Reporting system**: Separate policy, attractor-dominated, stabilized
3. **Structural symmetries**: Option order must be marginalized

The reporting system is not a readout of the action system. It is a learned policy that produces plausible-looking probability distributions, dominated by:

- **Primary attractor**: Uniform (25/25/25/25)
- **Secondary attractor**: Boundary hedge (when constrained)

## What These Findings Establish

1. Self-reported probabilities are unreliable indicators of model confidence
2. Belief-action decoupling is a capability limitation, not strategic avoidance
3. True logprobs are essential for confidence-related claims
4. Position debiasing via ensembling is a practical corrective method

## What These Findings Do NOT Claim

- That models have no uncertainty representations (they do, in the action system)
- That self-reports are useless (they reveal attractor structure)
- That all providers behave identically (they have different hedging strategies)
- That these effects transfer to non-multiple-choice formats
