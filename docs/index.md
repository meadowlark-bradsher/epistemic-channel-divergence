# Epistemic Channel Divergence

Research toolkit for measuring the divergence between what LLMs **believe** (behavioral/token-level) and what they **report** (declarative self-reports).

## Core Question

When asked to report their uncertainty, do language models give accurate probability estimates?

**Answer: No.** We find robust belief-action decoupling across providers, where models act confidently but report broad uncertainty—and this divergence is *largest* when the model is most confident.

## Key Findings

| Finding | Summary |
|---------|---------|
| **F1. Uniform Attractor** | Models default to 25/25/25/25 reports regardless of actual beliefs |
| **F2. Anti-Uniform Fix** | A mild constraint (5-80%) forces informative reports |
| **F3. Confidence-Divergence** | Mismatch is highest when models are most confident |
| **F4. Probe Order Matters** | Report→Act beats CoT→Act on ambiguous questions |
| **F5. Gravitational Pull** | Constrained models pile mass at boundaries |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/meadowlark-bradsher/epistemic-channel-divergence.git
cd epistemic-channel-divergence

# Set up API keys
echo "OPENAI_API_KEY=sk-..." > .env
echo "GOOGLE_API_KEY=..." >> .env

# Run belief probe experiment
python experiments/belief_probe_baseline.py --experiment2 --anti-uniform

# Run gravitational pull sweep
python experiments/gravitational_pull.py openai --multi-question
```

## Takeaway

LLMs exhibit a robust, learned decoupling between action confidence and reported uncertainty. Belief expression, reasoning, and action are partially independent control surfaces—measurable, intervenable, and not interchangeable.

## Navigation

- [Key Results](findings/index.md) - Summary of robust findings
- [Experiments](experiments/index.md) - How to run each experiment
- [Lab Notes](labnotes/belief-probe-notes.md) - Detailed research notes
