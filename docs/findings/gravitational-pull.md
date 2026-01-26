# Gravitational Pull

When constrained away from uniform reporting, models pile probability at constraint boundaries rather than expressing genuine beliefs.

## The Phenomenon

The uniform attractor isn't eliminated by constraints—it's **displaced**. When forced to break the 25/25/25/25 pattern, models don't distribute mass according to genuine beliefs. Instead, they report the closest permissible values to uniform.

## Evidence: Boundary Mass Under Tightening Constraints

| Constraint | OpenAI | Gemini | Llama 3.1 | Qwen 2.5 |
|------------|--------|--------|-----------|----------|
| (5, 80) | 17.9% | 28.4% | 9.9% | 18.4% |
| (10, 70) | 33.2% | 37.0% | 9.8% | 35.0% |
| (15, 60) | 32.2% | 29.6% | 8.6% | 38.1% |
| (20, 55) | **42.8%** | 30.6% | 16.2% | **37.8%** |

**Boundary mass** = fraction of probability allocated within 3% of min/max bounds.

## Gravitational Pull Strength by Provider

| Provider | Trend (5,80 → 20,55) | Strength |
|----------|---------------------|----------|
| OpenAI gpt-4o-mini | **+24.9%** | STRONG |
| Qwen 2.5 7B | **+19.3%** | STRONG |
| Llama 3.1 8B | +6.4% | MODERATE |
| Gemini 2.0-flash | +2.3% | WEAK |

## Provider-Specific Strategies

### Strong Pull (OpenAI, Qwen)

Boundary mass increases dramatically as constraints tighten. These models clearly displace hedging to constraint boundaries when squeezed.

```
Constraint   Boundary Mass (OpenAI)
(5,80)       ████░░░░░░░░░░░░░░░░  18%
(10,70)      ███████░░░░░░░░░░░░░  33%
(15,60)      ███████░░░░░░░░░░░░░  32%
(20,55)      █████████░░░░░░░░░░░  43%
                 ↑
          Mass piles at edges
```

### Moderate Pull (Llama)

Lower baseline boundary mass with modest increase. Less prone to boundary piling overall—may have different training distribution.

### Weak Pull (Gemini)

Starts with highest baseline but doesn't increase under pressure. Gemini hedges at boundaries **by default** rather than when forced.

Gemini also shows a persistent **uniform attractor** that resists the anti-uniform constraint:

| Provider | Uniform Rate (with constraint) |
|----------|-------------------------------|
| OpenAI | 0% |
| Gemini | 8-13% |

Gemini's "safe reporting" behavior is more deeply embedded.

## Interpretation

The gravitational pull effect reveals that:

1. **Constraints don't reveal genuine beliefs** - They just shift where hedging occurs
2. **Boundary piling is a "tell"** - High boundary mass indicates constrained hedging, not genuine uncertainty
3. **The attractor is displaced, not eliminated** - Models still want to hedge but can't report uniform
4. **Provider differences reflect training** - Different training regimes create different hedging strategies

## Implications

1. **Don't interpret boundary-heavy reports as calibrated** - They reflect constraint satisfaction, not belief
2. **Wider constraints are more informative** - Looser bounds reveal more genuine variation
3. **Provider comparison requires caution** - Different baseline strategies make cross-provider comparison difficult
4. **Multiple constraint levels needed** - Single constraint level can't distinguish real beliefs from displaced hedging

## Related

- [Uniform Attractor](uniform-attractor.md) - The primary phenomenon being displaced
- [Structural Sensitivity](structural-sensitivity.md) - Another lens on action vs report
