# Belief Probe Baseline

The foundational experiment for measuring belief-action divergence across different probe orderings.

## Hypothesis

Different orderings of action and belief elicitation affect alignment. Specifically:

- Report→Act (Probe B) should anchor action to declared beliefs
- CoT→Act (Probe C) should propagate reasoning to action
- Introspection (Probe D) should invoke different computational pathways

## What This Experiment Falsifies

- **Claim**: "Probe ordering is just presentation—it doesn't affect results"
- **Result**: Falsified. Probe B consistently shows lower JS divergence than Probe C.

- **Claim**: "CoT improves belief expression"
- **Result**: Falsified for ambiguous questions. CoT amplifies variance.

## What This Experiment Does NOT Claim

- That any probe provides "true" beliefs (all probes measure the same underlying decoupling)
- That these effects generalize beyond multiple-choice format
- That the optimal probe is universal (task-dependent)

## Probe Types

| Probe | Sequence | Mechanism |
|-------|----------|-----------|
| **A** | Act → Report | Post-hoc honesty |
| **B** | Report → Act | Belief anchoring |
| **C** | CoT → Act | Reasoning propagation |
| **D** | Act → Introspect | Retrospective framing |

## Usage

```bash
# Local Ollama baseline across all probes
python experiments/belief_probe_baseline.py llama3.1:latest --anti-uniform -n 50

# Multi-question experiment across all probes
python experiments/belief_probe_baseline.py llama3.1:latest --experiment2 --anti-uniform \
  -o data/experiment2_results.json

# Cross-provider comparison with true logprobs
python experiments/compare_providers.py --providers openai gemini --quick \
  -o data/provider_comparison.json
```

`belief_probe_baseline.py` is Ollama-oriented and estimates action probabilities by sampling.
For OpenAI, Gemini, or llama.cpp runs with true logprobs, use `experiments/compare_providers.py`.

### Options

| Flag | Description |
|------|-------------|
| `MODEL` | Ollama model name for `belief_probe_baseline.py` |
| `--anti-uniform` | Apply 5-80% constraint to break uniform attractor |
| `--experiment2` | Run multi-question experiment across all probes |
| `-n N` | Number of action samples for Ollama estimation |
| `--questions FILE` | Question set (defaults to `data/mc_questions.json`) |
| `-o FILE` | Output JSON file |

## Key Results

| Probe | Mean JS | Performance |
|-------|---------|-------------|
| B (Report→Act) | 0.28 | **Best** |
| D (Introspect) | 0.32 | Good |
| A (Act→Report) | 0.34 | Moderate |
| C (CoT→Act) | 0.38 | **Worst on ambiguous** |

### CoT by Question Type

| Category | CoT JS | Other Probes |
|----------|--------|--------------|
| Easy | 0.07 | ~0.10 |
| Ambiguous | 0.33 | ~0.28 |
| Adversarial | 0.32 | ~0.30 |

## Delivers

- F1-F2: Uniform attractor + anti-uniform effects
- F3: Belief-action decoupling
- F4: Probe order effects (Report→Act anchoring, CoT variance)
- CoT vs Introspect comparison

## Related

- [Findings: Probe Order Effects](../findings/probe-order-effects.md)
- [Findings: Belief-Action Decoupling](../findings/belief-action-decoupling.md)
