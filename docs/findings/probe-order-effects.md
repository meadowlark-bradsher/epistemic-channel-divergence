# Probe Order Effects

Probe order and framing are causal interventions, not presentation details.

## The Probe Types

| Probe | Sequence | Mechanism |
|-------|----------|-----------|
| **A** | Act → Report | Post-hoc honesty: Report after committing |
| **B** | Report → Act | Belief anchoring: Declare before acting |
| **C** | CoT → Act | Reasoning propagation: Reason, then act |
| **D** | Act → Introspect | Retrospective framing: Act, then reflect |

## Finding: Report→Act (Probe B) Produces Best Alignment

Forcing belief declaration *before* action consistently reduces mismatch:

| Probe | Mean JS | Effect |
|-------|---------|--------|
| B (Report→Act) | 0.28 | **Best** |
| D (Act→Introspect) | 0.32 | Good |
| A (Act→Report) | 0.34 | Moderate |
| C (CoT→Act) | 0.38 | **Worst on ambiguous** |

### Mechanism: Anchoring

When the model declares beliefs first, that declaration becomes an upstream latent variable. The subsequent action is conditioned on the stated belief, creating coupling via causal structure rather than introspection.

## Finding: CoT is a Variance Amplifier

Chain-of-thought has **regime-dependent effects**:

| Question Type | CoT Performance | JS Divergence |
|---------------|-----------------|---------------|
| Easy | Best | 0.07 |
| Ambiguous | Worst | 0.33 |
| Adversarial | Worst | 0.32 |

### Why CoT Fails on Ambiguous Questions

CoT reasoning commits to a narrative. When multiple plausible stories exist:

1. CoT selects one narrative path
2. The path becomes increasingly committed
3. Final action reflects the committed narrative
4. But no belief coherence propagates through the chain

CoT collapses **narratives**, not **uncertainty**. It doesn't improve belief expression—it amplifies variance by introducing an additional source of randomness.

### ESI Evidence

Under semantic-preserving perturbations, CoT shows highest action sensitivity:

| Provider | Probe C ESI_action | Other Probes |
|----------|-------------------|--------------|
| OpenAI | 0.297 | 0.161-0.194 |
| Gemini | 0.414 | 0.221-0.314 |

Small changes to question text propagate through the reasoning chain and amplify at the action point.

## Finding: Introspective Framing Stabilizes

Probe D (Act→Introspect) yields the most stable coupling across task types:

| Task Type | Probe D Variance | Other Probes |
|-----------|------------------|--------------|
| Mixed | Low | Higher |

### Mechanism: Regularization

Introspective framing ("reflect on what you just did") invokes a different computational pathway than direct reporting. It appears to regularize behavior, dampening the narrative collapse that CoT induces.

## Implications

1. **Framing matters causally** - Same question, different order = different alignment
2. **Don't use CoT for uncertainty** - It amplifies variance on ambiguous inputs
3. **Report→Act for best alignment** - Anchoring creates coupling
4. **Introspection for stability** - Less variance across task types

## Summary Table

| Goal | Recommended Probe |
|------|-------------------|
| Best alignment overall | B (Report→Act) |
| Most stable across tasks | D (Introspect) |
| Avoid on ambiguous | C (CoT) |

## Related

- [Belief-Action Decoupling](belief-action-decoupling.md) - The core phenomenon
- [Structural Sensitivity](structural-sensitivity.md) - Why CoT amplifies variance
