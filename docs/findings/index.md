# Key Results

This page summarizes the robust findings from our belief-action divergence experiments.

## F1. Uniform Probability Reporting is a Stable Learned Attractor

**Claim**: Absent constraints, models overwhelmingly report uniform distributions (25/25/25/25), regardless of how confident their actions are.

**Evidence**:

- 100% uniform reports across probes (A/B/C/D) without constraint
- Holds across providers and prompt orderings
- JSON compliance does not break the attractor

**Interpretation**: This is not ignorance or incapacity. It is a *policy equilibrium* induced by training and evaluation pressure. Self-report ≠ introspection by default.

---

## F2. A Mild Anti-Uniform Constraint Reliably Breaks the Attractor

**Claim**: Adding a gentle constraint (e.g., min 5%, max 80%) forces informative reports without inducing boundary gaming.

**Evidence**:

- Uniform reports drop from ~100% → ~0–2%
- No piling at min/max bounds (initially)
- Report entropy drops modestly (≈1.85–1.95), not collapse

**Interpretation**: Models *can* express graded uncertainty. The uniform attractor is optional, not fundamental.

---

## F3. Belief-Action Decoupling is Real and Largest When Action is Most Confident

**Claim**: With true logprobs, mismatch (JS divergence) is **highest when the action distribution is most peaked**.

**Evidence**:

- OpenAI / Gemini: Corr(JS, H_action) ≈ −0.5
- Divergence increases as entropy decreases
- Sampling-based estimates previously masked this effect

**Interpretation**: Models often act decisively while reporting broad uncertainty. This is classic belief-action decoupling, not noise.

---

## F4. Probe Order Matters

### F4a. Report→Act (Probe B) Acts as a Coupling Intervention

Forcing belief declaration *before* action generally reduces mismatch. The declared belief becomes an upstream latent variable—an anchoring device.

### F4b. Chain-of-Thought is a Variance Amplifier

CoT improves alignment on easy questions but **worsens** alignment on ambiguous/adversarial ones. When multiple plausible stories exist, CoT commits to one without propagating belief coherently.

### F4c. Introspective Framing Stabilizes Across Tasks

Introspective framing (Probe D) yields the most stable belief-action coupling on heterogeneous tasks. It *regularizes* behavior, dampening narrative collapse.

---

## F5. Gravitational Pull: The Uniform Attractor is Displaced, Not Eliminated

**Claim**: When constrained away from uniform (e.g., 5-80% bounds), models pile probability at constraint boundaries rather than expressing genuine interior beliefs.

**Evidence**:

| Provider | Boundary Mass: (5,80) → (20,55) | Pull Strength |
|----------|--------------------------------|---------------|
| OpenAI gpt-4o-mini | 18% → 43% | **STRONG** |
| Qwen 2.5 7B | 18% → 38% | **STRONG** |
| Llama 3.1 8B | 10% → 16% | MODERATE |
| Gemini 2.0-flash | 28% → 31% | WEAK |

**Interpretation**: The uniform attractor wasn't eliminated—just displaced to the boundaries. Models still want to hedge but can't report 25/25/25/25, so they report the closest permissible values.

---

## F6. Sampling-Based Action Estimation Can Invert Conclusions

**Claim**: Sampling with smoothing can reverse correlations and obscure true effects.

**Evidence**:

- Ollama (sampling): Corr(JS, H_action) positive
- OpenAI/Gemini (logprobs): Corr(JS, H_action) negative

**Interpretation**: True logprobs are essential for confidence-related claims. Sampling is acceptable for exploratory work but not final attribution.

---

## Mean JS Divergence Across Providers

| Provider | Mean JS |
|----------|---------|
| OpenAI gpt-4o-mini | 0.368 |
| Gemini 2.0-flash | 0.373 |
| Llama 3.1 8B | 0.374 |
| Qwen 2.5 7B | 0.466 |

All models show substantial belief-action misalignment (~0.37-0.47 JS divergence).
