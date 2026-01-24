# Epistemic Channel Divergence

Research toolkit for measuring the divergence between what LLMs **believe** (behavioral/token-level) and what they **report** (declarative self-reports).

## Core Question

When asked to report their uncertainty, do language models give accurate probability estimates?

**Answer: No.** We find robust belief-action decoupling across providers, where models act confidently but report broad uncertainty—and this divergence is *largest* when the model is most confident.

## Key Findings

### F1. Uniform Reporting Attractor
Without constraints, models overwhelmingly report uniform distributions (25/25/25/25) regardless of their actual token probabilities. This is a learned "safe" policy, not ignorance.

### F2. Anti-Uniform Constraint Breaks It
A mild constraint (5-80% per option) forces informative reports without inducing gaming. The uniform attractor is optional, not fundamental.

### F3. Belief-Action Decoupling Increases with Confidence
With true logprobs, mismatch (JS divergence) is **highest when action is most peaked**. Models act decisively while reporting broad uncertainty.

### F4. Probe Order Matters
- **Report→Act (Probe B)**: Best alignment—declared belief anchors subsequent action
- **CoT→Act (Probe C)**: Variance amplifier—helps easy questions, hurts ambiguous ones
- **Act→Introspect (Probe D)**: Most stable across heterogeneous tasks

### F5. Gravitational Pull
When constrained away from uniform, models pile probability at constraint boundaries rather than expressing genuine interior beliefs. The uniform attractor is displaced, not eliminated.

## Experiments

### Belief Probe Baseline
Compare behavioral vs declarative beliefs across four probe orderings:

| Probe | Order | Purpose |
|-------|-------|---------|
| A | Act → Report | Post-hoc honesty |
| B | Report → Act | Belief anchoring |
| C | CoT → Act | Reasoning propagation |
| D | Act → Introspect | Retrospective report |

```bash
# Single probe run
python experiments/belief_probe_baseline.py llama3.1:latest --anti-uniform -n 50

# Multi-question experiment
python experiments/belief_probe_baseline.py --experiment2 --anti-uniform -o data/results.json
```

### Cross-Provider Comparison
Compare OpenAI, Gemini, and local models:

```bash
python experiments/compare_providers.py --providers openai gemini --quick
```

### Gravitational Pull
Test whether constraint tightening increases boundary piling:

```bash
# Single question sweep
python experiments/gravitational_pull.py openai --n-trials 10 --verbose

# Multi-question sweep across providers
python experiments/gravitational_pull.py openai --multi-question -o data/gp_results.json
```

### Alignment Regularizer
Test whether calibration incentives improve belief reporting:

```bash
python experiments/alignment_regularizer.py openai
```

## Providers

The toolkit supports multiple inference backends:

| Provider | Token Beliefs | Setup |
|----------|---------------|-------|
| OpenAI | True logprobs | `OPENAI_API_KEY` in `.env` |
| Gemini | True logprobs | `GOOGLE_API_KEY` in `.env` |
| llama.cpp | True logprobs | Run `llama-server -m model.gguf -c 4096` |
| Ollama | Sampling-based | `ollama serve` (less accurate) |

**Note:** True logprobs are essential for confidence-related claims. Sampling-based estimation can invert correlations.

## Project Structure

```
experiments/
├── belief_probe_baseline.py    # Core probe harness
├── compare_providers.py        # Cross-provider comparison
├── gravitational_pull.py       # Constraint sweep experiments
├── alignment_regularizer.py    # Calibration incentive tests
├── generate_mc_questions.py    # Question generation
├── providers.py                # Provider abstraction layer
└── test_gemini.py              # Provider smoke test

data/
├── mc_questions.json           # 40 test questions (easy/ambiguous/adversarial)
├── gp_mq_*.json               # Gravitational pull results by provider
├── provider_comparison.json    # Cross-provider experiment results
└── experiment2_results.json    # Multi-question probe results

docs/labnotes/
├── belief-probe-notes.md                  # Probe experiment notes
├── gravitational-pull-experiment.md       # Constraint sweep analysis
└── token-belief-vs-stated-belief.md       # Synthesis of findings
```

## Setup

1. Create `.env` with API keys:
   ```
   OPENAI_API_KEY=sk-...
   GOOGLE_API_KEY=...
   ```

2. For local models, install llama.cpp:
   ```bash
   brew install llama.cpp

   # Download a model (e.g., Llama 3.1 8B)
   huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
       --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
       --local-dir models/

   # Start server
   llama-server -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -c 4096
   ```

3. No dependencies beyond Python stdlib (uses `urllib` for API calls).

## Metrics

- **JS Divergence**: Jensen-Shannon divergence between token distribution and reported probabilities
- **Boundary Mass**: Fraction of reported probability within 3% of constraint bounds
- **Report Entropy**: Shannon entropy of reported distribution
- **Token Entropy**: Shannon entropy of behavioral (action) distribution

## Results Summary

### Provider Comparison (Gravitational Pull)

| Provider | Boundary Mass (5,80) → (20,55) | Pull Strength |
|----------|-------------------------------|---------------|
| OpenAI gpt-4o-mini | 18% → 43% | **STRONG** |
| Qwen 2.5 7B | 18% → 38% | **STRONG** |
| Llama 3.1 8B | 10% → 16% | MODERATE |
| Gemini 2.0-flash | 28% → 31% | WEAK |

### Mean JS Divergence (Belief-Action Misalignment)

| Provider | Mean JS |
|----------|---------|
| OpenAI gpt-4o-mini | 0.368 |
| Gemini 2.0-flash | 0.373 |
| Llama 3.1 8B | 0.374 |
| Qwen 2.5 7B | 0.466 |

All models show substantial belief-action misalignment (~0.37-0.47 JS divergence).

## Takeaway

LLMs exhibit a robust, learned decoupling between action confidence and reported uncertainty. Belief expression, reasoning, and action are partially independent control surfaces—measurable, intervenable, and not interchangeable.
