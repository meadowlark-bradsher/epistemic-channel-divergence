# Belief-Action Decoupling

A deep dive into the phenomenon where LLMs act confidently but report broad uncertainty.

## What is Belief-Action Decoupling?

When we probe an LLM's uncertainty, we can measure two distinct signals:

1. **Behavioral Belief**: The model's action distribution (token softmax / logprobs over answer choices)
2. **Reported Belief**: The model's explicit probability report when asked to quantify uncertainty

Belief-action decoupling occurs when these two signals disagree. A model might:

- **Act decisively** (90% probability on option A)
- **Report hedged** (25% on each option)

## Measuring Decoupling

We use **Jensen-Shannon divergence** to quantify the mismatch:

\[
JS(P \| Q) = \frac{1}{2} KL(P \| M) + \frac{1}{2} KL(Q \| M)
\]

where \(M = \frac{1}{2}(P + Q)\).

JS divergence ranges from 0 (identical distributions) to 1 (maximally different). In our experiments:

| JS Range | Interpretation |
|----------|----------------|
| 0.0 - 0.1 | Good alignment |
| 0.1 - 0.3 | Moderate mismatch |
| 0.3 - 0.5 | Substantial decoupling |
| > 0.5 | Severe decoupling |

## The Confidence-Divergence Relationship

A key finding: **decoupling is worst when the model is most confident**.

With true logprobs (OpenAI, Gemini), we observe:

```
Corr(JS_divergence, H_action) ≈ -0.5
```

This means:

- Low action entropy → High divergence
- High action entropy → Lower divergence

When a model is uncertain (high entropy), its reports are more aligned with behavior. When confident (low entropy), it hedges in reports while acting decisively.

## Why Does This Happen?

Several hypotheses:

### 1. Training Pressure
RLHF and evaluation may reward cautious probability reports. Appearing overconfident could be penalized during training.

### 2. Lack of Introspective Access
Models may not have direct access to their own token probabilities when generating self-reports. They generate "plausible-looking" distributions rather than genuine introspection.

### 3. Different Circuits
Action selection and probability reporting may use partially separate computational pathways, leading to natural divergence.

## Evidence Against Strategic Hedging

We tested whether models *could* align but *choose* not to by adding calibration incentives:

| Incentive Level | Effect on JS |
|-----------------|--------------|
| None | 0.35 |
| Weak ("reports should reflect beliefs") | 0.35 |
| Moderate ("scoring system will compare") | 0.35 |
| Strong ("CALIBRATION CHECK") | 0.35 |

**No effect.** Models produce identical reports regardless of incentive, suggesting this is a capability limitation rather than a strategic choice.

## Implications

1. **Self-reported probabilities are unreliable** - Don't trust LLM confidence estimates at face value
2. **Action distributions are more informative** - Logprobs reveal actual model state better than self-reports
3. **Probe design matters** - Report→Act ordering reduces decoupling via anchoring
4. **True logprobs are essential** - Sampling-based estimation masks the effect

## Further Reading

- [Gravitational Pull Experiment](../experiments/gravitational-pull.md) - How constraints displace but don't eliminate hedging
- [Belief Probe Baseline](../experiments/belief-probe.md) - The core experimental framework
- [Token Belief vs Stated Belief](../labnotes/token-belief-vs-stated-belief.md) - Comprehensive synthesis
