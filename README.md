# Epistemic Channel Divergence

**What LLMs say they believe ≠ what their actions reveal.**

When you ask a language model "How confident are you?", the answer comes from a learned reporting policy, not introspective access to internal states. This repository provides tools to measure the divergence between three distinct channels: what models **do** (action distributions), what they **say** (self-reports), and what they **think they're doing** (reasoning traces).

## The Core Finding

Models can fall back to uniform probability reports (25/25/25/25) while simultaneously taking confident actions. The constrained and cross-provider artifacts are checked in today; the unconstrained paper-grade baseline is now wired up for regeneration but is not yet committed.

**Implications**: 
- Self-reported calibration metrics may be measuring report quality, not belief quality
- Belief elicitation requires constrained prompts that break the uniform attractor
- Action distributions (via logprobs) reveal confidence that self-reports obscure

## What's Here

```
epistemic-channel-divergence/
├── experiments/       # Runnable experiment scripts
├── interventions/     # Perturbation and debiasing utilities
├── analysis/          # Notebook + reusable analysis code for paper tables/figures
├── data/              # Checked-in question sets and JSON outputs
├── results/           # Checked-in JSONL / CSV outputs
├── docs/              # Findings, methods, experiment pages, lab notes
└── illustrations/     # Figure notes and illustration scripts
```

## Quick Start

```bash
# Setup
git clone https://github.com/meadowlark-bradsher/epistemic-channel-divergence.git
cd epistemic-channel-divergence
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Fill in any provider keys you plan to use
# OPENAI_API_KEY=...
# GEMINI_API_KEY=...

# Local Ollama baseline (sampling-based action estimates)
python experiments/belief_probe_baseline.py llama3.1:latest --experiment2 --anti-uniform \
  -o data/experiment2_results.json

# Unconstrained baseline for auditing the uniform-reporting claim
python experiments/belief_probe_baseline.py llama3.1:latest --experiment2 \
  -o data/unconstrained_baseline_results.json

# Cross-provider comparison (true logprobs where available)
python experiments/compare_providers.py --providers openai gemini --quick \
  -o data/provider_comparison.json

# ESI / structural sensitivity run
python experiments/esi_runner.py --providers openai gemini --full \
  -o results/esi/full_experiment.jsonl

# Reproduce the paper pipeline from the checked-in question set
make paper-pipeline

# Generate the unconstrained F1 baseline artifact without running the full pipeline
make unconstrained-baseline

# Recompute paper tables/figures from committed artifacts only
make analysis-summary
make analysis-notebook-check

# Run a fresh end-to-end pipeline including question generation
# (requires ANTHROPIC_API_KEY or CLAUDE_API_KEY)
make fresh-pipeline

# Run regression tests
make test
```

## Key Findings

**F1**: Historical local runs suggest unconstrained reporting collapses toward uniform reports regardless of action confidence  
**F2**: Mild anti-uniform constraints break the attractor (low-single-digit uniform rates in the checked-in baseline)  
**F3**: Divergence is *highest* when models are most confident (Corr(JS, H_action) ≈ -0.5)  
**F4**: Probe order matters - Report→Act anchors and reduces divergence  
**F5**: True logprobs essential - sampling-based estimation can invert correlations  
**F6**: CoT amplifies variance on ambiguous questions (→ near-zero correlation)  
**F7**: Introspection acts as a stabilizer across task types  

→ [Full findings with evidence and strength ratings](docs/findings/index.md)

The unconstrained raw artifact behind F1 is not yet checked in. Use `make unconstrained-baseline`
to generate `data/unconstrained_baseline_results.json`; future baseline outputs now record
parse-health and uniform-rate summaries both with and without parse failures included.

## For Researchers

**If you care about LLM calibration or uncertainty quantification:**
- [Findings](docs/findings/index.md) → What we know with high confidence
- [Methods](docs/methods/methods.md) → Why certain measurement choices matter
- Cites needed: belief elicitation, calibration metrics, UQ in LLMs

**If you're working on the Bridge Experiment or entropy coupling:**
- This characterizes what's *in* token entropy (H_tok) before measuring flow
- Self-reports may be thermodynamically inert - action distributions are where epistemic work happens
- See: connection to H_tok ↔ H_sec coupling hypothesis

## For Practitioners

**If you need reliable uncertainty estimates from LLMs:**
- Don't trust naive self-reports - they default to hedging
- Use constrained elicitation (see [anti-uniform constraints](docs/methods/methods.md#mitigating-reporting-biases))
- Extract action distributions via logprobs when available
- Ensemble over probe orderings to marginalize position bias

**If you're building with LLM APIs:**
- OpenAI: Use `logprobs` for action distributions
- Gemini: Extract via direct API access to token probabilities
- llama.cpp: Use the OpenAI-compatible server with `logprobs`
- Ollama in this repo is sampling-based only, so treat it as a baseline / fallback

## Running Experiments

Each experiment targets specific findings and runs in <10 minutes:

```bash
# F1: Unconstrained baseline for the uniform-reporting claim
python experiments/belief_probe_baseline.py llama3.1:latest --experiment2 \
  -o data/unconstrained_baseline_results.json

# F2: Anti-uniform follow-up
python experiments/belief_probe_baseline.py llama3.1:latest --experiment2 --anti-uniform \
  -o data/experiment2_results.json

# F3-F5: Cross-provider comparison + JS/entropy correlation
python experiments/compare_providers.py --providers openai gemini --quick \
  -o data/provider_comparison.json

# F5: Gravitational pull under tighter constraints
python experiments/gravitational_pull.py -p openai --multi-question \
  -o data/gp_results_openai.json

# F6 & F7: CoT, introspection, SOC, and order sensitivity
python experiments/esi_runner.py --providers openai gemini --full \
  -o results/esi/full_experiment.jsonl

# Ensemble measurement / position debiasing
python experiments/esi_ensemble.py --providers openai gemini --bootstrap \
  -o results/esi/ensemble_bootstrap.jsonl
```

Checked-in artifacts already live in `data/` and `results/esi/`. The new analysis layer is split:
`analysis/figures_and_tables.ipynb` is notebook-only and reads committed local artifacts, while
the generation layer remains in `experiments/`. The post-hoc ESI script
(`experiments/esi_analysis.py`) is still available for script-first workflows.

## Reproducibility

- `requirements.txt` captures the Python dependencies needed to run experiments and build docs.
- `.env.example` shows the provider variables used by the current scripts.
- `Makefile` provides `paper-pipeline`, `fresh-pipeline`, `unconstrained-baseline`, and `test` entrypoints.
- `analysis/figures_and_tables.ipynb` is the reviewer-facing notebook for reproducing tables/figures from committed data only.
- `make analysis-notebook-check` executes that notebook headlessly via `nbconvert`, which is suitable for CI sanity checks.
- OpenAI and Gemini defaults are snapshot-pinned in code: `gpt-4o-mini-2024-07-18` and `gemini-2.0-flash-001`.
- Question sets and several saved experiment outputs are committed under `data/`.
- [data/README.md](data/README.md) documents the checked-in JSON artifacts and their provenance.
- Future `belief_probe_baseline.py` JSON outputs now include parse-health summaries, raw report text, and per-probe action/report probabilities for audit.
- ESI raw outputs and summaries are committed under `results/esi/`.
- [results/README.md](results/README.md) documents the checked-in ESI/ensemble artifacts.
- The repo uses a hybrid model: scripts in `experiments/` generate artifacts, and `analysis/` turns committed artifacts into paper tables and figures.

## Architecture: The Three-Layer Model

Everything reduces to three layers with two attractors:

**Layer 1: Action System**
- Operates on token distributions  
- Sensitive to position bias, framing, reasoning paths
- Reveals confidence via logprobs (when available)
- Unstable: 3-6x more variance than reports under perturbations

**Layer 2: Reporting System**  
- Learned policy for expressing uncertainty
- Dominated by uniform hedge (primary) and boundary hedge (secondary)
- Largely insensitive to incentives, calibration threats, small perturbations
- Stabilized by RLHF pressure toward cautious expressions

**Layer 3: Structural Symmetries**
- Multiple-choice option order is a nuisance variable
- Single-shot measurement samples one arbitrary embedding
- Argmax flips in 52-64% of orderings (position bias alone)
- Ensembling over orderings (K≈4) marginalizes symmetry

→ [See methods for full treatment](docs/methods/methods.md)

## Measurement Principles

**What works:**
- True logprobs for action distributions (not sampling estimates)
- Anti-uniform constraints for breaking reporting attractors  
- Ensemble measurement over probe orderings (K=4-6)
- JS divergence for quantifying belief-report mismatch

**What doesn't:**
- Naive self-reports (uniform attractor dominates)
- Sampling-based action estimates (introduces smoothing bias)
- Single-shot measurement (confounded by position bias)
- Incentive-based elicitation (reporting policy is pre-trained)

→ [Methods document](docs/methods/methods.md) explains why each matters

## Status

**Stable / checked-in**:
- F2-F5: Core divergence phenomena with committed artifacts
- Anti-uniform constraint effectiveness
- Logprobs vs sampling distinction

**Pending artifact recommit**:
- F1 unconstrained baseline for the uniform-reporting claim

**Active Development**:
- Provider-specific calibration curves
- Adaptive probe selection policies  
- Belief revision sequences (Act→Report→Act₂)
- Connection to environmental ground truth

**Open Questions**:
- Is the reporting policy architecturally distinct or contextually triggered?
- Can belief revision be controlled (vs mere confabulation)?
- How does internal alignment relate to external accuracy?

## Citation

```bibtex
@software{bradsher2025epistemic,
  title={Epistemic Channel Divergence: Measuring Belief-Action-Report Decoupling in LLMs},
  author={Bradsher, Meadowlark},
  year={2025},
  url={https://github.com/meadowlark-bradsher/epistemic-channel-divergence}
}
```

## Navigation

- **[Findings](docs/findings/index.md)** - Authoritative claims with evidence levels
- **[Methods](docs/methods/)** - Experimental design and measurement principles  
- **[Experiments](docs/experiments/index.md)** - Runnable code for each finding
- **[Analysis](analysis/README.md)** - Notebook and reusable code for paper tables/figures
- **[Lab Notes](docs/labnotes/)** - Research provenance and key discoveries

## Contact

Questions, replications, or extensions? Open an issue or reach out.

---

*This work is part of broader research on entropy coupling in language models (the Bridge Experiment). See: [meadowlark-bradsher.github.io](https://meadowlark-bradsher.github.io)*
