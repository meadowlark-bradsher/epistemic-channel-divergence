# Belief-Action Decoupling

Models act confidently but report broad uncertainty—and this divergence is largest when the model is most confident.

## The Phenomenon

When we probe an LLM's uncertainty, two distinct signals emerge:

| Channel | Source | Example |
|---------|--------|---------|
| **Action** | Token softmax / logprobs | 90% on option A |
| **Report** | Explicit probability statement | 25% on each option |

Belief-action decoupling occurs when these signals disagree. A model might act decisively (90% on A) while reporting hedged uncertainty (25% each).

## The Confidence-Divergence Relationship

The key finding: **decoupling is worst when the model is most confident**.

With true logprobs (OpenAI, Gemini):

```
Corr(JS_divergence, H_action) ≈ -0.5
```

| Action Entropy      | Mean JS Divergence | Interpretation      |
| ------------------- | ------------------ | ------------------- |
| Low (<1 bit)        | ~0.45              | Severe decoupling   |
| Medium (1-1.5 bits) | ~0.35              | Moderate decoupling |
| High (>1.5 bits)    | ~0.25              | Lower decoupling    |

When uncertain (high entropy), reports align better with behavior. When confident (low entropy), models hedge in reports while acting decisively.

## Measuring Decoupling

We use **Jensen-Shannon divergence** to quantify mismatch:

$$JS(P \| Q) = \frac{1}{2} KL(P \| M) + \frac{1}{2} KL(Q \| M)$$

where $M = \frac{1}{2}(P + Q)$.

| JS Range | Interpretation |
|----------|----------------|
| 0.0 - 0.1 | Good alignment |
| 0.1 - 0.3 | Moderate mismatch |
| 0.3 - 0.5 | Substantial decoupling |
| > 0.5 | Severe decoupling |

## Mean Divergence by Provider

| Provider | Mean JS | Notes |
|----------|---------|-------|
| OpenAI gpt-4o-mini | 0.368 | True logprobs |
| Gemini 2.0-flash | 0.373 | True logprobs |
| Llama 3.1 8B | 0.374 | True logprobs (llama.cpp) |
| Qwen 2.5 7B | 0.466 | True logprobs (llama.cpp) |

All models show substantial belief-action misalignment (~0.37-0.47 JS divergence).

## Why Does This Happen?

### Hypothesis 1: Training Pressure (Supported)

RLHF and evaluation may reward cautious probability reports. Appearing overconfident could be penalized during training, creating a stable "safe reporting" policy.

### Hypothesis 2: Strategic Hedging (Not Supported)

We tested whether models *could* align but *choose* not to:

| Incentive | Effect on JS |
|-----------|--------------|
| None | 0.35 |
| "Reports should reflect beliefs" | 0.35 |
| "Scoring system will compare" | 0.35 |
| "CALIBRATION CHECK" | 0.35 |

**No effect.** This is a capability limitation, not strategic avoidance.

### Hypothesis 3: Separate Circuits (Consistent)

Action selection and probability reporting may use partially separate computational pathways. The "reporting circuit" doesn't have direct access to the "action circuit."

## Implications

1. **Self-reported probabilities are unreliable** - Don't trust LLM confidence estimates at face value
2. **Action distributions are more informative** - Logprobs reveal actual model state better than self-reports
3. **True logprobs are essential** - Sampling-based estimation can invert correlations and mask the effect
4. **Probe design matters** - Report→Act ordering reduces decoupling via anchoring

## Related

- [Probe Order Effects](probe-order-effects.md) - How framing affects alignment
- [Structural Sensitivity](structural-sensitivity.md) - Differential sensitivity of the two channels
