# Structural Sensitivity

Action distributions are 3-6x more sensitive to semantic-preserving perturbations than reported beliefs.

## The Phenomenon

When we apply semantic-preserving interventions to prompts, the action (token) and report (stated) channels respond differently:

- **Action**: Highly sensitive, distributions shift dramatically
- **Report**: Largely stable, distributions remain similar

This asymmetry reveals that the two channels have fundamentally different properties.

## ESI: Epistemic Sensitivity under Intervention

We measure sensitivity using **Hellinger distance** between distributions before and after intervention:

$$H(P, Q) = \frac{1}{\sqrt{2}} \sqrt{\sum_i (\sqrt{p_i} - \sqrt{q_i})^2}$$

| Metric | Definition |
|--------|------------|
| ESI_action | Mean Hellinger distance of action distributions under intervention |
| ESI_report | Mean Hellinger distance of report distributions under intervention |
| Ratio | ESI_action / ESI_report |

## Interventions Tested

### SOC (Skip-One-Char)

Remove characters from word tails while preserving semantic content:

- "programming" → "programing"
- "understanding" → "understading"

### ORDER (Option Position Shuffling)

Shuffle A/B/C/D option positions while preserving content-to-label mapping:

- Original: A=Paris, B=London, C=Berlin, D=Madrid
- Shuffled: A=Berlin, B=Paris, C=Madrid, D=London

## Results

### SOC Intervention

| Provider | Probe | ESI_action | ESI_report | Ratio |
|----------|-------|------------|------------|-------|
| OpenAI | A | 0.185 | 0.037 | 5.0x |
| OpenAI | B | 0.161 | 0.048 | 3.4x |
| OpenAI | C | **0.297** | 0.048 | **6.2x** |
| OpenAI | D | 0.194 | 0.063 | 3.1x |
| Gemini | A | 0.221 | 0.080 | 2.8x |
| Gemini | B | 0.314 | 0.079 | 4.0x |
| Gemini | C | **0.414** | 0.068 | **6.1x** |
| Gemini | D | 0.258 | 0.069 | 3.7x |

### ORDER Intervention

| Provider | Probe | ESI_action | ESI_report | Ratio |
|----------|-------|------------|------------|-------|
| OpenAI | All | ~0.53 | ~0.09 | ~6x |
| Gemini | All | ~0.50 | ~0.10 | ~5x |

## Key Findings

### F1. Action is 3-6x More Sensitive Than Report

Across all conditions:

- **SOC**: Action ~0.2-0.4, Report ~0.04-0.08
- **ORDER**: Action ~0.5, Report ~0.09-0.12

The reporting policy acts as a **stabilizer** that produces consistent outputs even when the underlying action distribution shifts dramatically.

### F2. Option Order Causes Massive Action Shifts

ORDER intervention causes ~0.5 Hellinger distance—argmax flips in 52-64% of orderings.

This reveals strong **position bias**: the A/B/C/D labels carry information beyond the content they label.

### F3. CoT Amplifies Action Sensitivity

CoT (Probe C) shows highest ESI_action:

| Provider | CoT ESI_action | Other Probes |
|----------|----------------|--------------|
| OpenAI | 0.297 | 0.161-0.194 |
| Gemini | 0.414 | 0.221-0.314 |

Chain-of-thought makes actions **more brittle**, not less. Small prompt changes propagate through the reasoning chain and amplify at the action point.

### F4. Report Stability is Remarkably Consistent

ESI_report stays in a narrow range (0.04-0.12) regardless of:

- Intervention type (SOC vs ORDER)
- Probe type (A, B, C, D)
- Provider (OpenAI vs Gemini)

The reporting policy is highly regularized—decoupled from the instability of the action channel.

### F5. High-Entropy Actions Are More Stable

| Entropy | N | Mean ESI_action |
|---------|---|-----------------|
| Low (<1 bit) | 252 | 0.393 |
| Mid (1-1.5 bits) | 85 | 0.409 |
| High (>1.5 bits) | 57 | 0.343 |

Counter-intuitively, uncertain actions are slightly **more stable** than confident ones. Peaked distributions may be more sensitive to small perturbations.

## Interpretation

The asymmetry between action and report sensitivity confirms the **two-system model**:

1. **Action system**: Moment-to-moment computation, sensitive to surface features
2. **Reporting system**: Learned policy, trained to produce stable outputs

The report channel doesn't read out the action channel—it runs a separate, regularized process.

## Implications

1. **Token probabilities are unstable signals** - Claims about "model confidence" from single measurements are unreliable
2. **Repeated measurement needed** - Perturbation-based robustness checks are essential
3. **Position debiasing is critical** - ~0.5 Hellinger from order alone demands ensembling
4. **CoT increases fragility** - Don't assume reasoning improves reliability

## Related

- [Measurement Implications](measurement-implications.md) - How to correct for these effects
- [Probe Order Effects](probe-order-effects.md) - Why CoT amplifies variance
